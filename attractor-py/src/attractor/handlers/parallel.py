# src/attractor/handlers/parallel.py
"""Parallel execution handlers for fan-out and fan-in operations."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..models import Graph, Context, Node, StageStatus
from .interface import Handler
from . import CodergenBackend


@dataclass
class BranchResult:
    """Result from a single parallel branch."""

    branch_id: str
    node_id: str
    status: StageStatus
    outcome: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ParallelResults:
    """Serialized results stored in context."""

    branches: List[BranchResult] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    total_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "node_id": b.node_id,
                    "status": b.status.value,
                    "error": b.error,
                }
                for b in self.branches
            ],
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_count": self.total_count,
        }


class ParallelHandler(Handler):
    """Fan-out handler for concurrent branch execution.

    Executes multiple branches concurrently, each with an isolated
    context clone. Results are stored for downstream fan-in.
    """

    def __init__(self, llm_backend: Optional[CodergenBackend] = None):
        self._llm_backend = llm_backend

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: str,
    ) -> Any:
        from ..engine.executor import PipelineExecutor, ExecutorConfig
        from ..models import Outcome
        import os

        # Get configuration
        join_policy = node.attrs.get("join_policy", "wait_all")
        error_policy = node.attrs.get("error_policy", "continue")
        max_parallel = int(node.attrs.get("max_parallel", "4"))

        # Find all outgoing edges - these are the parallel branches
        branches = []
        for edge in graph.edges:
            if edge.from_node == node.id:
                branches.append(edge)

        if not branches:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No outgoing branches found for parallel node",
            )

        # Create stage directory
        stage_dir = os.path.join(logs_root, node.id)
        os.makedirs(stage_dir, exist_ok=True)

        # Execute branches with bounded parallelism
        results = self._execute_branches(
            branches=branches,
            node=node,
            context=context,
            graph=graph,
            logs_root=logs_root,
            stage_dir=stage_dir,
            max_parallel=max_parallel,
            error_policy=error_policy,
        )

        # Count results
        success_count = sum(1 for r in results if r.status == StageStatus.SUCCESS)
        failure_count = sum(1 for r in results if r.status == StageStatus.FAIL)

        # Store results in context
        parallel_results = ParallelResults(
            branches=results,
            success_count=success_count,
            failure_count=failure_count,
            total_count=len(results),
        )

        context_updates = {"parallel.results": parallel_results.to_dict()}

        # Determine outcome based on join policy
        if join_policy == "first_success" and success_count > 0:
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates=context_updates,
                notes=f"First success policy: {success_count}/{len(results)} branches succeeded",
            )
        elif join_policy == "any_success" and success_count > 0:
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates=context_updates,
                notes=f"Any success policy: {success_count}/{len(results)} branches succeeded",
            )
        elif join_policy == "wait_all":
            if success_count == len(results):
                return Outcome(
                    status=StageStatus.SUCCESS,
                    context_updates=context_updates,
                    notes=f"All {len(results)} branches succeeded",
                )
            else:
                return Outcome(
                    status=StageStatus.PARTIAL_SUCCESS if success_count > 0 else StageStatus.FAIL,
                    context_updates=context_updates,
                    notes=f"Wait all policy: {success_count}/{len(results)} branches succeeded",
                )

        return Outcome(
            status=StageStatus.SUCCESS if success_count > 0 else StageStatus.FAIL,
            context_updates=context_updates,
        )

    def _execute_branches(
        self,
        branches: List[Any],
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: str,
        stage_dir: str,
        max_parallel: int,
        error_policy: str,
    ) -> List[BranchResult]:
        """Execute branches with bounded parallelism."""
        from ..models import Outcome
        from ..engine.executor import PipelineExecutor, ExecutorConfig

        results = []
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_branch(branch, index):
            async with semaphore:
                # Clone context for this branch
                branch_context = context.clone()

                # Create sub-executor for this branch
                config = ExecutorConfig(
                    logs_root=logs_root,
                    llm_backend=self._llm_backend,
                )
                executor = PipelineExecutor(config)

                try:
                    # Execute the branch starting from the target node
                    result = self._execute_branch_sync(
                        executor=executor,
                        start_node=branch.to_node,
                        graph=graph,
                        context=branch_context,
                        logs_root=logs_root,
                    )

                    return BranchResult(
                        branch_id=f"{node.id}_branch_{index}",
                        node_id=branch.to_node,
                        status=result.status,
                        outcome=result,
                    )

                except Exception as e:
                    if error_policy == "fail_fast":
                        raise
                    return BranchResult(
                        branch_id=f"{node.id}_branch_{index}",
                        node_id=branch.to_node,
                        status=StageStatus.FAIL,
                        error=str(e),
                    )

        # Run async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                asyncio.gather(*[execute_branch(b, i) for i, b in enumerate(branches)])
            )
        finally:
            loop.close()

        return results

    def _execute_branch_sync(
        self,
        executor: Any,
        start_node: str,
        graph: Graph,
        context: Context,
        logs_root: str,
    ) -> Any:
        """Execute a single branch synchronously."""
        from ..models import Outcome

        # Get the node to execute
        node = graph.nodes.get(start_node)
        if not node:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Branch target node not found: {start_node}",
            )

        # Execute just this node (not the full subgraph for simplicity)
        handler = executor._handler_registry.resolve(node)
        return handler.execute(node, context, graph, logs_root)


class FanInHandler(Handler):
    """Fan-in handler for consolidating parallel results.

    Reads parallel.results from context and selects the best candidate
    based on an optional LLM evaluation prompt.
    """

    def __init__(self, llm_backend: Optional[CodergenBackend] = None):
        self._llm_backend = llm_backend

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: str,
    ) -> Any:
        from ..models import Outcome

        # Read parallel results
        results_dict = context.get("parallel.results")
        if not results_dict:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No parallel results to evaluate",
            )

        # Parse results
        branches = results_dict.get("branches", [])
        success_count = results_dict.get("success_count", 0)

        if not branches:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No branches in parallel results",
            )

        # If there's a prompt, use LLM to select the best result
        if node.prompt and self._llm_backend:
            best = self._evaluate_with_llm(
                node=node, context=context, branches=branches, logs_root=logs_root
            )
        else:
            # Default: select first successful branch
            successful = [b for b in branches if b.get("status") == "success"]
            if successful:
                best = successful[0]
            else:
                best = branches[0]

        # Record winner in context
        context_updates = {
            "parallel.fan_in.best_id": best.get("branch_id", ""),
            "parallel.fan_in.best_node": best.get("node_id", ""),
        }

        return Outcome(
            status=StageStatus.SUCCESS if success_count > 0 else StageStatus.FAIL,
            context_updates=context_updates,
            notes=f"Selected branch {best.get('branch_id')} as best result",
        )

    def _evaluate_with_llm(
        self,
        node: Node,
        context: Context,
        branches: List[Dict],
        logs_root: str,
    ) -> Dict:
        """Use LLM to evaluate and select the best branch result."""
        # For simplicity, return first successful branch
        # A full implementation would construct a prompt with all results
        # and ask the LLM to select the best one
        successful = [b for b in branches if b.get("status") == "success"]
        if successful:
            return successful[0]
        return branches[0]
