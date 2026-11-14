import sys
import os
import argparse

# ensure repo root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import MLOpsAgentOrchestrator


def main():
    parser = argparse.ArgumentParser(description="MLOps AI Agent CLI")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
        help="Natural language prompt for the agent"
    )
    args = parser.parse_args()

    orchestrator = MLOpsAgentOrchestrator()
    print("=" * 60)
    print(f"User Query: {args.query}")
    print("=" * 60)

    response = orchestrator.process_query(args.query)

    print("\n--- AGENT ANSWER ---")
    print(response.answer)

    print("\n--- KNOWN FACTS ---")
    for fact in response.known_facts:
        print(f" [FACT] {fact}")

    print("\n--- UNCERTAINTIES ---")
    if response.uncertainties:
        for unc in response.uncertainties:
            print(f" [UNCERTAINTY] {unc}")
    else:
        print(" None (all evidence verified)")

    print("\n--- RECOMMENDED ACTIONS ---")
    for act in response.recommended_actions:
        print(f" [ACTION] {act}")

    print(f"\n--- RETRAINING RECOMMENDED: {response.retraining_recommended} ---")
    if response.retraining_justification:
        print(f" Reason: {response.retraining_justification}")

    print("\n--- EVIDENCE COLLECTED ---")
    for ev in response.evidence:
        status_str = "SUCCESS" if ev.success else "FAILED"
        print(f" - {ev.tool_name} [{ev.category}] -> {status_str} ({ev.duration_ms}ms)")

    print("=" * 60)


if __name__ == "__main__":
    main()
