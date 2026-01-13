"""
IKYKE Protocol Demo

Demonstrates the IKYKE automated workflow protocol.
"""

from .ikyke_format import IkykeFileFormat
from .ikyke_protocol import IkykeProtocol
from .logic_layer import LogicEngine
from .data_layer import DataLayer
import time


def demo_ikyke_workflow():
    """Demonstrate IKYKE workflow creation and execution."""
    print("=" * 70)
    print("IKYKE Protocol Demo")
    print("=" * 70)
    print()
    
    # Create a workflow
    print("1. Creating IKYKE workflow...")
    workflow = IkykeFileFormat.create_default("Demo Experiment")
    
    # Configure for quick demo (30 seconds instead of 3-5 minutes)
    workflow.run.duration_min = 0.5  # 30 seconds
    workflow.run.duration_max = 0.5
    workflow.auto_save.interval_seconds = 10
    workflow.run.max_formulas = 10  # Limit for demo
    
    # Add test formulas
    workflow.formulas = [
        "And(x, y)",
        "Or(x, Not(y))",
        "Implies(x, y)",
        "And(Or(x, y), Not(z))",
        "Or(And(x, y), And(Not(x), z))"
    ]
    
    print(f"   Workflow: {workflow.name}")
    print(f"   Formulas: {len(workflow.formulas)}")
    print(f"   Duration: {workflow.run.duration_min}-{workflow.run.duration_max} minutes")
    print()
    
    # Save workflow
    print("2. Saving workflow...")
    workflow_path = IkykeFileFormat.save(workflow, "demo_workflow.ikyke")
    print(f"   Saved to: {workflow_path}")
    print()
    
    # Create protocol engine
    print("3. Creating protocol engine...")
    logic_engine = LogicEngine()
    data_layer = DataLayer()
    
    protocol = IkykeProtocol(
        workflow=workflow,
        logic_engine=logic_engine,
        data_layer=data_layer,
        container_path="demo_container.ikyke_container"
    )
    
    # Set up callbacks
    def on_phase_change(phase):
        print(f"   Phase changed: {phase.value}")
    
    def on_formula_result(result):
        status = "SAT" if result.satisfiable else "UNSAT"
        print(f"   Formula: {result.formula[:40]}... -> {status} ({result.solve_time:.4f}s)")
    
    protocol.on_phase_change = on_phase_change
    protocol.on_formula_result = on_formula_result
    print()
    
    # Run workflow
    print("4. Running workflow...")
    print("   (This will run for 30 seconds)")
    print()
    
    protocol.start()
    
    # Monitor progress
    while protocol.running:
        status = protocol.get_status()
        print(f"   Status: {status['phase']} | Formulas: {status['formula_count']} | "
              f"Time: {status['elapsed_time']:.1f}s")
        time.sleep(5)
    
    # Wait for completion
    if protocol.thread:
        protocol.thread.join()
    
    print()
    print("5. Workflow completed!")
    print()
    
    # Display results
    container = protocol.container
    
    if container.evaluation_results:
        eval_res = container.evaluation_results
        print("Evaluation Results:")
        print(f"   Total Formulas: {eval_res.total_formulas}")
        print(f"   Satisfiable: {eval_res.satisfiable_count}")
        print(f"   Unsatisfiable: {eval_res.unsatisfiable_count}")
        print(f"   Satisfiability Rate: {eval_res.satisfiability_rate:.2%}")
        print(f"   Total Models: {eval_res.total_models}")
        print(f"   Average Solve Time: {eval_res.avg_solve_time:.4f}s")
        print()
    
    if container.query_results:
        print("Query Results:")
        for qr in container.query_results:
            print(f"   Query: {qr.query}")
            print(f"   Results: {qr.count} formulas")
            print()
    
    if container.analysis_result:
        print("Analysis:")
        if container.analysis_result.report:
            print(container.analysis_result.report[:500])
        print()
    
    print("=" * 70)
    print("Demo completed!")
    print(f"Workflow file: {workflow_path}")
    print(f"Container file: demo_container.ikyke_container")
    print("=" * 70)


if __name__ == "__main__":
    demo_ikyke_workflow()
