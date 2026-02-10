# src/attractor/server.py
"""HTTP server mode for remote pipeline execution."""
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .models import Graph, StageStatus
from .parser import parse_dot
from .validator import validate
from .engine import PipelineExecutor, ExecutorConfig


class PipelineState(Enum):
    """Pipeline execution states."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineRun:
    """A single pipeline execution run."""

    run_id: str
    graph: Graph
    state: PipelineState = PipelineState.IDLE
    result: Optional[Any] = None
    current_node: Optional[str] = None
    pending_question: Optional[Dict] = None
    human_answer: Optional[str] = None
    logs_root: str = "./logs"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ServerConfig:
    """HTTP server configuration."""

    host: str = "localhost"
    port: int = 8080
    logs_root: str = "./logs"
    llm_backend: Optional[Any] = None
    interviewer_factory: Optional[Callable] = None


class PipelineRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for pipeline operations."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data: Dict, status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, message: str, status: int = 400) -> None:
        """Send error response."""
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/health":
            self._handle_health()
        else:
            self._send_error("Not found", 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/run":
            self._handle_run()
        elif self.path == "/answer":
            self._handle_answer()
        else:
            self._send_error("Not found", 404)

    def _handle_health(self) -> None:
        """Handle health check."""
        self._send_json({"status": "ok"})

    def _handle_status(self) -> None:
        """Handle status check."""
        server = self.server.pipeline_server
        run_id = self._get_run_id()

        if run_id not in server.runs:
            self._send_error("Run not found", 404)
            return

        run = server.runs[run_id]

        self._send_json({
            "run_id": run_id,
            "state": run.state.value,
            "current_node": run.current_node,
            "pending_question": run.pending_question,
            "result": self._serialize_result(run.result) if run.result else None,
            "error": run.error,
            "created_at": run.created_at,
            "completed_at": run.completed_at,
        })

    def _handle_run(self) -> None:
        """Handle pipeline run request."""
        server = self.server.pipeline_server

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode()) if body else {}

            dot_source = data.get("dot")
            if not dot_source:
                self._send_error("Missing 'dot' field")
                return

            # Parse and validate
            graph = parse_dot(dot_source)
            diagnostics = validate(graph)
            errors = [d for d in diagnostics if d.severity.value == "error"]

            if errors:
                self._send_json({
                    "error": "Validation failed",
                    "diagnostics": [d.to_dict() for d in diagnostics],
                }, 400)
                return

            # Create run
            import uuid
            run_id = str(uuid.uuid4())

            logs_root = Path(server.config.logs_root) / run_id
            logs_root.mkdir(parents=True, exist_ok=True)

            run = PipelineRun(
                run_id=run_id,
                graph=graph,
                logs_root=str(logs_root),
            )
            server.runs[run_id] = run

            # Start execution in background thread
            thread = threading.Thread(
                target=self._execute_pipeline,
                args=(run, server),
            )
            thread.daemon = True
            thread.start()

            self._send_json({
                "run_id": run_id,
                "state": "running",
            })

        except Exception as e:
            self._send_error(str(e), 500)

    def _handle_answer(self) -> None:
        """Handle human answer submission."""
        server = self.server.pipeline_server
        run_id = self._get_run_id()

        if run_id not in server.runs:
            self._send_error("Run not found", 404)
            return

        run = server.runs[run_id]

        if run.state != PipelineState.WAITING_FOR_HUMAN:
            self._send_error(f"Run not waiting for human input (state: {run.state.value})")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode()) if body else {}

            answer = data.get("answer")
            if answer is None:
                self._send_error("Missing 'answer' field")
                return

            run.human_answer = answer

            self._send_json({"status": "ok"})

        except Exception as e:
            self._send_error(str(e), 500)

    def _get_run_id(self) -> str:
        """Extract run_id from query parameters."""
        from urllib.parse import urlparse, parse_qs

        query = parse_qs(urlparse(self.path).query)
        return query.get("run_id", [None])[0]

    def _serialize_result(self, result: Any) -> Dict:
        """Serialize execution result."""
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return {
            "status": str(result),
        }

    def _execute_pipeline(self, run: PipelineRun, server: "PipelineServer") -> None:
        """Execute pipeline in background thread."""
        try:
            run.state = PipelineState.RUNNING

            # Create interviewer that waits for HTTP answers
            interviewer = server.config.interviewer_factory() if server.config.interviewer_factory else None

            config = ExecutorConfig(
                logs_root=run.logs_root,
                llm_backend=server.config.llm_backend,
                interviewer=interviewer,
            )

            executor = PipelineExecutor(config)
            result = executor.run(run.graph)

            run.result = result
            run.state = PipelineState.COMPLETED if result.status == StageStatus.SUCCESS else PipelineState.FAILED
            run.completed_at = time.time()

        except Exception as e:
            run.state = PipelineState.FAILED
            run.error = str(e)
            run.completed_at = time.time()


class PipelineServer:
    """HTTP server for pipeline execution."""

    def __init__(self, config: ServerConfig):
        """Initialize the pipeline server.

        Args:
            config: Server configuration
        """
        self.config = config
        self.runs: Dict[str, PipelineRun] = {}
        self._server: Optional[HTTPServer] = None

    def start(self) -> None:
        """Start the HTTP server."""
        handler = PipelineRequestHandler
        self._server = HTTPServer((self.config.host, self.config.port), handler)
        self._server.pipeline_server = self

        print(f"Starting pipeline server on http://{self.config.host}:{self.config.port}")
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get a pipeline run by ID."""
        return self.runs.get(run_id)


def create_default_server(
    host: str = "localhost",
    port: int = 8080,
    logs_root: str = "./logs",
    llm_backend: Optional[Any] = None,
) -> PipelineServer:
    """Create a pipeline server with default configuration.

    Args:
        host: Server host
        port: Server port
        logs_root: Root directory for logs
        llm_backend: Optional LLM backend

    Returns:
        Configured pipeline server
    """
    config = ServerConfig(
        host=host,
        port=port,
        logs_root=logs_root,
        llm_backend=llm_backend,
    )
    return PipelineServer(config)
