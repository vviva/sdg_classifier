from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json
import re
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_sdg(row, confidence_cutoff=0.8):
    """
    Classify a project into one or more SDGs using OpenAI API.
    Returns a list of SDG numbers, target codes, and confidence score.
    Only returns predictions with confidence >= confidence_cutoff.
    """
    title = row.get("project_title", "")
    short_desc = row.get("short_description", "")
    long_desc = row.get("long_description", "")
    donor = row.get("donor_name", "")
    agency = row.get("agency_name", "")
    purpose = row.get("purpose_name", "")
    sector = row.get("sector_name", "")

    # Combine all fields into one detailed project description
    project_text = f"""
    Project Title: {title}
    Short Description: {short_desc}
    Long Description: {long_desc}
    Donor: {donor}
    Implementing Agency: {agency}
    Purpose: {purpose}
    Sector: {sector}
    """

    user_prompt = f"""
    You are an SDG classification expert.
    Based on the full project information below (including donor, agency, purpose, and sector),
    assign the project to one or more Sustainable Development Goals (SDGs) and, if possible,
    identify the most relevant targets.
    Use the donor, agency, purpose, and sector fields to help infer the most relevant SDG(s).

    Example: If agency is WHO or UNFPA, the project is likely Goal 3 (health-related).
    If agency is FAO, likely Goal 2 (food security/agriculture), etc.
.
    Do NOT include goals that are only weakly related.
    If the project does not contribute clearly to any SDG, return {{"sdg_goals": [], "confidence": 0}}.

    Return ONLY a JSON object (no markdown, no extra text) with this exact format:
    {{"sdg_goals": [list of SDG numbers], "targets": [list of target codes like "3.1", "8.5"], "confidence": number between 0 and 1}}

    Example: {{"sdg_goals": [1, 3, 8], "targets": ["1.4", "3.8", "8.5"], "confidence": 0.85}}

    Title: {title}
    Short description: {short_desc}
    Long description: {long_desc}
    """

    # Combine system and user prompts for the Responses API
    combined_prompt = f"""You are an expert on SDG classification. Return only valid JSON.

{user_prompt}"""

    response = client.responses.create(
        model="gpt-4-mini",
        input=combined_prompt
    )

    try:
        content = response.output_text.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        confidence = result.get("confidence", 0)

        # Filter by confidence cutoff
        if confidence >= confidence_cutoff:
            return result.get("sdg_goals", []), result.get("targets", []), confidence
        else:
            return [], [], confidence
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Response was: {response.output_text[:200] if 'response' in locals() else 'No response'}")
        return [], [], 0


def safe_parse_sdg_code(x):
    """Parse sdg_code column like '805|806' or 'DNC'."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x.upper() == "DNC":
            return []
        # Split by pipe and filter out empties
        return [v.strip() for v in x.split("|") if v.strip() and v.upper() != "DNC"]
    return []


def extract_goals_from_targets(target_list):
    """
    Extract goal numbers from target-level SDGs.
    Example: 501 -> 5, 1302 -> 13, 805 -> 8
    """
    goals = []
    for t in target_list:
        t_str = str(t).strip()
        # Skip non-numeric values
        if not t_str or not t_str.isdigit():
            continue
        # Example: 501 -> 5, 1302 -> 13
        if len(t_str) > 2:
            goals.append(int(t_str[:-2]))
        else:
            goals.append(int(t_str))
    return sorted(set(goals))  # remove duplicates


def extract_goals_from_predictions(pred_list):
    """
    Convert predictions (list of integers) to sorted set.
    """
    goals = []
    for p in pred_list:
        if isinstance(p, int) and 1 <= p <= 17:
            goals.append(p)
        elif isinstance(p, str):
            # Try to extract number from string like "SDG3" or "3"
            match = re.search(r"\d+", p)
            if match:
                num = int(match.group())
                if 1 <= num <= 17:
                    goals.append(num)
    return sorted(set(goals))


def any_match(true, pred):
    """Check if at least one goal matches."""
    return len(set(true) & set(pred)) > 0


def evaluate_classifier(n_samples=10, input_file="ground_truth.csv", output_file="evaluation_results_openai.csv",
                        confidence_cutoff=0.8):
    """
    Evaluate the OpenAI classifier on a small subset of ground truth data.
    Uses the same metrics as accuracy_check.py:
    - Exact match accuracy
    - At least one goal correct

    Args:
        n_samples: number of samples to evaluate (default: 10)
        input_file: input CSV file path (default: ground_truth.csv)
        output_file: output CSV file path (default: evaluation_results_openai.csv)
        confidence_cutoff: minimum confidence threshold (default: 0.8)
    """
    print(f"Loading ground truth data from: {input_file}")
    df = pd.read_csv(input_file, nrows=n_samples)

    print(f"Evaluating on {len(df)} samples...")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        # Get inputs
        title = row.get("project_title", "")
        short_desc = row.get("short_description", "")
        long_desc = row.get("long_description", "")

        # Skip if all inputs are missing
        if pd.isna(title) and pd.isna(short_desc) and pd.isna(long_desc):
            continue

        # Replace NaN with empty string for display
        title = "" if pd.isna(title) else str(title)
        short_desc = "" if pd.isna(short_desc) else str(short_desc)
        long_desc = "" if pd.isna(long_desc) else str(long_desc)

        # Get ground truth from sdg_code column
        sdg_code_raw = row.get("sdg_code", "")
        sdg_targets = safe_parse_sdg_code(sdg_code_raw)
        true_goals = extract_goals_from_targets(sdg_targets)

        # Get prediction
        pred_sdgs, pred_targets, confidence = classify_sdg(row, confidence_cutoff=confidence_cutoff)
        pred_goals = extract_goals_from_predictions(pred_sdgs)

        # Calculate matches
        exact = set(true_goals) == set(pred_goals)
        any_match_result = any_match(true_goals, pred_goals) if true_goals else False

        results.append({
            "project_title": title,
            "short_description": short_desc,
            "long_description": long_desc,
            "sdg_code": sdg_code_raw,
            "true_goals": true_goals,
            "pred_goals": pred_goals,
            "pred_targets": pred_targets,
            "confidence": confidence,
            "exact_match": exact,
            "any_match": any_match_result
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Calculate overall metrics (same as accuracy_check.py)
    exact_match_acc = results_df["exact_match"].mean()
    any_match_acc = results_df["any_match"].mean()

    print(f"\nExact match accuracy: {exact_match_acc:.2%}")
    print(f"At least one goal correct: {any_match_acc:.2%}")

    # Show detailed results for each sample
    print("\n" + "=" * 80)
    print("Detailed Results per Sample:")
    print("=" * 80)
    for i, result in results_df.iterrows():
        title_display = result['project_title'][:100] if len(result['project_title']) > 100 else result['project_title']
        print(f"\n{i + 1}. {title_display}")
        print(f"   SDG Code: {result['sdg_code']}")
        print(f"   True Goals: {result['true_goals']}")
        print(f"   Pred Goals: {result['pred_goals']}")
        print(f"   Pred Targets: {result['pred_targets']}")
        print(f"   Exact Match: {result['exact_match']}, Any Match: {result['any_match']}")
        print(f"   Confidence: {result['confidence']}")

    # Save results to file
    results_df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate OpenAI SDG classifier using Responses API')
    parser.add_argument('--samples', '-n', type=int, default=10,
                        help='Number of samples to evaluate (default: 10)')
    parser.add_argument('--input', '-i', type=str, default='ground_truth.csv',
                        help='Input CSV file path (default: ground_truth.csv)')
    parser.add_argument('--output', '-o', type=str, default='evaluation_results_openai.csv',
                        help='Output CSV file path (default: evaluation_results_openai.csv)')
    parser.add_argument('--confidence', '-c', type=float, default=0.8,
                        help='Minimum confidence threshold (default: 0.8)')

    args = parser.parse_args()

    print(f"Starting evaluation with {args.samples} samples...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Confidence cutoff: {args.confidence}")
    results = evaluate_classifier(n_samples=args.samples, input_file=args.input, output_file=args.output,
                                  confidence_cutoff=args.confidence)
    print("\nEvaluation complete!")
