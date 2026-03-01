"""
CLI entry point for Petal.

Usage:
    petal eval evals/weather-agent.yaml
    petal eval  (runs all .yaml files in evals/)
"""

from __future__ import annotations

import glob
import sys

import click

from petal.eval.runner import EvalResult, run_eval
from petal.eval.scorers import answer_relevance, cost_scorer, tool_accuracy


# Registry of all built-in scorers
BUILT_IN_SCORERS = [tool_accuracy, cost_scorer, answer_relevance]


@click.group()
def main():
    """Petal — Agent framework with built-in evals."""
    pass


@main.command()
@click.argument("eval_path", required=False)
@click.option("--ci", is_flag=True, help="Exit with code 1 if any test fails")
def eval(eval_path: str | None, ci: bool):
    """Run eval test cases against an agent."""

    # Find eval files
    if eval_path:
        paths = [eval_path]
    else:
        paths = sorted(glob.glob("evals/**/*.yaml", recursive=True))
        if not paths:
            click.echo("No eval files found in evals/")
            sys.exit(1)

    # We need the user to provide an agent — for now, the CLI demonstrates
    # the eval runner format. In practice, users import their agent in a
    # config file or pass a module path. For the MVP, we show a helpful error.
    click.echo("Note: CLI eval requires an agent. Use run_eval() in Python for now.")
    click.echo(f"Found {len(paths)} eval file(s): {', '.join(paths)}")
    click.echo()
    click.echo("Example usage in Python:")
    click.echo("  from petal import Agent")
    click.echo("  from petal.eval.runner import run_eval")
    click.echo("  from petal.eval.scorers import tool_accuracy, cost_scorer")
    click.echo("  result = run_eval('evals/my-eval.yaml', agent=my_agent, scorers=[tool_accuracy, cost_scorer])")


def print_results(result: EvalResult) -> bool:
    """Print eval results as a formatted table. Returns True if all passed."""

    # Calculate column widths
    name_width = max(len(c.name) for c in result.cases) if result.cases else 10
    name_width = max(name_width, 9)  # "Test Case" header

    # Collect all scorer names from the results
    scorer_names = []
    if result.cases:
        scorer_names = list(result.cases[0].scores.keys())

    # Header
    header = f"  {'Test Case':<{name_width}}"
    for sn in scorer_names:
        header += f"  {sn:>12}"
    header += f"  {'Cost':>8}"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    # Rows
    all_passed = True
    for case in result.cases:
        row = f"  {case.name:<{name_width}}"
        for sn in scorer_names:
            score = case.scores.get(sn, 0.0)
            threshold = case.thresholds.get(sn)
            if threshold is not None:
                status = "PASS" if score >= threshold else "FAIL"
                row += f"  {score:.2f} {status:>4}"
            else:
                row += f"  {score:>12.2f}"
        row += f"  ${case.cost:>7.4f}"

        if not case.passed:
            all_passed = False

        click.echo(row)

    # Summary
    click.echo()
    click.echo(f"  Overall: {result.passed}/{len(result.cases)} passed")
    click.echo(f"  Duration: {result.duration_ms:.0f}ms")

    return all_passed


if __name__ == "__main__":
    main()
