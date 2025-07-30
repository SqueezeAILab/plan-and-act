#!/usr/bin/env python3

import argparse
import json
import os


def get_screenshots(path: str) -> dict[str, list[str]]:
    """
    Get screenshots from the given path.

    Args:
        path: Path to the screenshots directory

    Returns:
        dictionary mapping task_id to list of screenshot paths
    """
    screenshots = {}
    if not os.path.exists(path):
        print(f"Warning: Screenshots path {path} does not exist")
        return screenshots

    dirs = sorted(os.listdir(path))
    for dir in dirs:
        task_id = dir
        if os.path.isdir(os.path.join(path, dir)):
            files = sorted(os.listdir(os.path.join(path, dir)))
            for file in files:
                if file.endswith(".png"):
                    if task_id not in screenshots:
                        screenshots[task_id] = []
                    screenshots[task_id].append(os.path.join(path, dir, file))
    return screenshots


def get_urls(path: str) -> dict[str, str]:
    """
    Get URLs from the test set configuration files.

    Args:
        path: Path to the test set configuration files

    Returns:
        dictionary mapping task_id to URL
    """
    urls = {}
    if not os.path.exists(path):
        print(f"Warning: Test set path {path} does not exist")
        return urls

    test = sorted(os.listdir(path))
    for task in test:
        try:
            with open(os.path.join(path, task), "r") as f:
                config = json.load(f)
                urls[task] = config["start_url"]
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not parse URL from {task}: {e}")
    return urls


def get_evals(path: str) -> dict[str, str]:
    """
    Get evaluation data from the test set configuration files.

    Args:
        path: Path to the test set configuration files

    Returns:
        dictionary mapping task_id to evaluation data
    """
    evals = {}
    if not os.path.exists(path):
        print(f"Warning: Test set path {path} does not exist")
        return evals

    test = sorted(os.listdir(path))
    for task in test:
        try:
            with open(os.path.join(path, task), "r") as f:
                config = json.load(f)
                evals[task] = config["eval"]
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not parse eval data from {task}: {e}")
    return evals


def generate_markdown_report(
    experiment_data: list[dict],
    urls: dict[str, str] | None = None,
    eval_data: dict[str, str] | None = None,
    task_ids: list[str] | None = None,
    exclude_task_ids: list[str] | None = None,
) -> str:
    """
    Generate a markdown report for the experiment data.

    Args:
        experiment_data: list of experiment data dictionaries
        urls: dictionary mapping task_id to URL
        eval_data: dictionary mapping task_id to evaluation data
        task_ids: list of task IDs to filter by (if None, include all)
        exclude_task_ids: list of task IDs to exclude from the report (if not specified, all tasks are included)

    Returns:
        Markdown string
    """
    markdown = ""
    n_incorrect = 0

    for experiment in experiment_data:
        # Skip if task_ids is provided and this experiment's task_id is not in the list
        if task_ids and experiment["task_id"] not in task_ids:
            continue

        if exclude_task_ids and experiment["task_id"] in exclude_task_ids:
            continue

        if experiment["score"] == 0:
            n_incorrect += 1
            markdown += f"# Experiment Analysis\n\n"
            markdown += f"**Experiment:** {experiment['experiment_name']} ({experiment['task_id']})\n\n"
            markdown += f"**Site:** {experiment['site']}\n\n"

            if urls and experiment["task_id"] in urls:
                markdown += f"**URL:** {urls[experiment['task_id']]}\n\n"

            # markdown += f"**Global Plan:**\n```\n{experiment['global_plan']}\n```\n\n"
            markdown += f"**Score:** {experiment['score']}\n\n"

            if eval_data and experiment["task_id"] in eval_data:
                markdown += f"**Eval:**\n```\n{json.dumps(eval_data[experiment['task_id']], indent=2)}\n```\n\n"

            markdown += "**Actions:**\n\n```\n"
            for action in experiment["actions"]:
                if isinstance(action, str):
                    action_str = action.replace("\n", " ")
                    # TODO: Later make this an argument
                    action_str = (
                        action_str.split("[Start of Action]")[1]
                        .split("[End of Action]")[0]
                        .strip()
                    )
                elif isinstance(action, dict):
                    action_str = action["action_str"].replace("\n", " ")
                else:
                    raise ValueError(f"Unknown action type: {type(action)}")

                markdown += f"- {action_str}\n"
            markdown += "```\n\n---\n\n"

    markdown += f"\nNumber of incorrect experiments: {n_incorrect}\n"
    return markdown


def generate_html_report(
    experiment_data: list[dict],
    screenshot_data: dict[str, list[str]] | None = None,
    urls: dict[str, str] | None = None,
    eval_data: dict[str, str] | None = None,
    task_ids: list[str] | None = None,
    exclude_task_ids: list[str] | None = None,
) -> str:
    """
    Generate an HTML report for the experiment data with optional screenshots.

    Args:
        experiment_data: list of experiment data dictionaries
        screenshot_data: dictionary mapping task_id to list of screenshot paths
        urls: dictionary mapping task_id to URL
        eval_data: dictionary mapping task_id to evaluation data
        task_ids: list of task IDs to filter by (if None, include all)
        exclude_task_ids: list of task IDs to exclude from the report (if not specified, all tasks are included)

    Returns:
        HTML string
    """
    n_incorrect = 0
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
        img { max-width: 45vw; border: 1px solid #ddd; border-radius: 4px; }
        .image-container { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
        .image-wrapper { margin-bottom: 20px; }
        hr { margin: 30px 0; border: 0; border-top: 1px solid #ddd; }
        h1 { color: #333; }
        h2 { color: #444; }
        h3 { color: #555; }
    </style>
</head>
<body>
"""

    for experiment in experiment_data:
        # Skip if task_ids is provided and this experiment's task_id is not in the list
        if task_ids and experiment["task_id"] not in task_ids:
            continue

        if exclude_task_ids and experiment["task_id"] in exclude_task_ids:
            continue

        if experiment["score"] == 0:
            n_incorrect += 1
            html_content += f"<h1>Experiment Analysis</h1>"
            html_content += f"<h2>Experiment: {experiment['experiment_name']} ({experiment['task_id']})</h2>"
            html_content += f"<p><strong>Site:</strong> {experiment['site']}</p>"

            if urls and experiment["task_id"] in urls:
                html_content += f"<p><strong>URL:</strong> <a href='{urls[experiment['task_id']]}'>{urls[experiment['task_id']]}</a></p>"

            html_content += (
                f"<h3>Global Plan:</h3><pre>{experiment['global_plan']}</pre>"
            )
            html_content += f"<p><strong>Score:</strong> {experiment['score']}</p>"

            if eval_data and experiment["task_id"] in eval_data:
                html_content += f"<h3>Eval:</h3><pre>{json.dumps(eval_data[experiment['task_id']], indent=2)}</pre>"

            html_content += "<h3>Actions:</h3><pre>"
            for action in experiment["actions"]:
                action_str = action.replace("\n", " ")
                html_content += f"- {action_str}<br>"
            html_content += "</pre>"

            # Add images if available
            if screenshot_data:
                task_id = experiment["task_id"].split(".")[0]
                if task_id in screenshot_data and screenshot_data[task_id]:
                    html_content += (
                        "<h3>Trajectory Images:</h3><div class='image-container'>"
                    )
                    for shot in screenshot_data[task_id]:
                        html_content += f"<div class='image-wrapper'><img src='{shot}' alt='Screenshot'></div>"
                    html_content += "</div>"

            html_content += "<hr>"

    html_content += (
        f"<p><strong>Number of incorrect experiments:</strong> {n_incorrect}</p>"
    )
    html_content += "</body></html>"

    return html_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate error analysis report from experiment data"
    )
    parser.add_argument("--input_path", help="Path to the score data JSON file")
    parser.add_argument(
        "--output",
        "-o",
        default="experiment_analysis",
        help="Output file path (without extension)",
    )
    parser.add_argument(
        "--screenshots",
        "-s",
        action="store_true",
        help="Include screenshots in the report",
    )
    parser.add_argument(
        "--screenshots_path", default=None, help="Path to the screenshots directory"
    )
    parser.add_argument(
        "--test_set_path", default=None, help="Path to the test set configuration files"
    )
    parser.add_argument(
        "--format",
        choices=["html", "markdown", "both"],
        default="html",
        help="Output format",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        help="List of task IDs to include in the report (if not specified, all tasks are included)",
    )
    parser.add_argument(
        "--exclude_task_ids",
        nargs="+",
        help="List of task IDs to exclude from the report (if not specified, all tasks are included)",
    )

    args = parser.parse_args()

    # Load experiment data
    try:
        with open(args.input_path, "r") as f:
            score_data = json.load(f)
            experiment_data = score_data.get("experiment_specific_score_data", [])
            if not experiment_data:
                print(f"Warning: No experiment data found in {args.input_path}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading experiment data: {e}")
        return

    # Get URLs and eval data if test_set_path is provided
    urls = None
    eval_data = None
    if args.test_set_path:
        urls = get_urls(args.test_set_path)
        eval_data = get_evals(args.test_set_path)

    # Get screenshots if requested
    screenshot_data = None
    if args.screenshots and args.screenshots_path:
        screenshot_data = get_screenshots(args.screenshots_path)

    # Generate reports based on format
    if args.format in ["markdown", "both"]:
        markdown = generate_markdown_report(
            experiment_data,
            urls,
            eval_data,
            args.task_ids,
            args.exclude_task_ids,
        )
        markdown_path = f"{args.output}.md"
        with open(markdown_path, "w") as f:
            f.write(markdown)
        print(f"Markdown report saved to {markdown_path}")

    if args.format in ["html", "both"]:
        html = generate_html_report(
            experiment_data,
            screenshot_data,
            urls,
            eval_data,
            args.task_ids,
            args.exclude_task_ids,
        )
        html_path = f"{args.output}.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    main()
