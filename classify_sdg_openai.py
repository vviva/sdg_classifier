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


def classify_sdg(row):
    """
    Classify a project into SDG targets using OpenAI API.
    Returns SDG codes (e.g., 301 for Goal 3 Target 1, 300 for Goal 3 only) and confidence score.
    Only returns classifications with individual confidence >= 0.8.
    """
    title = row.get("ProjectTitle", "")
    short_desc = row.get("ShortDescription", "")
    long_desc = row.get("LongDescription", "")
    donor = row.get("DonorName", "")
    agency = row.get("AgencyName", "")
    purpose = row.get("PurposeName", "")
    sector = row.get("SectorNamee", "")

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
    You are an expert in the United Nations Sustainable Development Goals (SDGs).

    Your task is to classify development projects to the **most relevant SDG targets**
    using the **official UN SDG target definitions** (as published by the United Nations).
    Use the donor, agency, purpose, and sector fields to help infer the most relevant SDG(s).

    Example: If agency is WHO or UNFPA, the project is likely Goal 3 (health-related).
    If agency is FAO, likely Goal 2 (food security/agriculture), etc.

    ---
    **INSTRUCTIONS**
    1. Carefully analyze the project information below.
    2. Compare it with the official SDG targets (for all 17 goals).
    3. Assign the project to **one or more SDG targets** that are most directly relevant.
    4. If the project is **vague**, **general**, or focused on **coordination, management, or administration**
       without a direct thematic focus (e.g., "humanitarian coordination", "oversight", "co-financing",
       "logistics support", "program management"), then return an empty classifications list.
       - Avoid forcing a classification when there is no clear substantive focus.
    5. Return the results in a valid JSON object.
    6. If the project clearly does not relate to any SDG target, but it can be clearly related at the goal level,
       return a goal number (1-17) with an empty targets list.
    7. If the project clearly does not relate to any SDG target nor a goal, return an empty classifications list.
    8. IMPORTANT: The "goal" field must ALWAYS be a number (1-17), never text like "DNC".

    ---
    **OUTPUT FORMAT (strict JSON)**
    {{
      "classifications": [
        {{
          "goal": goal number (e.g. 5),
          "targets": [list of target codes for this goal, e.g. ["5.1", "5.2"]],
          "confidence": float between 0 and 1 for this specific goal
        }}
      ]
    }}

    ---
    **Example responses:**

    Example 1 (with specific targets):
    {{
      "classifications": [
        {{
          "goal": 5,
          "targets": ["5.1", "5.2"],
          "confidence": 0.95
        }},
        {{
          "goal": 10,
          "targets": ["10.2"],
          "confidence": 0.82
        }}
      ]
    }}

    Example 2 (goal level only, no specific targets):
    {{
      "classifications": [
        {{
          "goal": 3,
          "targets": [],
          "confidence": 0.85
        }}
      ]
    }}

    Example 3 (no clear SDG connection):
    {{
      "classifications": []
    }}

    ---
    **Project Information:**
    {project_text}

    Use only the official UN SDG target definitions when making your decision.
    """
    response = None
    try:
        # Combine system and user prompts for the Responses API
        combined_prompt = f"""You are a highly accurate SDG classification model that maps projects to official UN SDG targets.

{user_prompt}"""

        response = client.responses.create(
            model="gpt-4o",
            input=combined_prompt
        )
        content = response.output_text

        # Extract JSON safely - handle cases where JSON might be malformed
        # First, try to fix common issues like unquoted DNC
        content_fixed = content.replace('"goal": DNC', '"goal": "DNC"')

        json_match = re.search(r"\{.*\}", content_fixed, re.S)
        if not json_match:
            return [], 0

        try:
            result = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # If JSON parsing still fails, return empty
            return [], 0

        # Parse new format with per-goal confidence scores
        classifications = result.get("classifications", [])

        # Filter classifications with confidence >= 0.8 and convert to SDG codes
        sdg_codes = []
        all_confidences = []

        for item in classifications:
            conf = item.get("confidence", 0)
            all_confidences.append(conf)
            if conf >= 0.8:
                goal = item.get("goal")

                # Skip if goal is DNC or not a valid number
                if goal in ["DNC", "dnc", None] or not isinstance(goal, (int, float)):
                    continue

                # Convert goal to int if it's a float
                try:
                    goal = int(goal)
                except (ValueError, TypeError):
                    continue

                targets = item.get("targets", [])
                if goal:
                    if targets:
                        # If targets are specified, create codes like 301, 302, etc.
                        for target in targets:
                            # Extract target number (e.g., "3.1" -> 1, "3.12" -> 12)
                            target_num = target.split('.')[-1] if '.' in str(target) else target
                            try:
                                sdg_code = goal * 100 + int(target_num)
                                sdg_codes.append(sdg_code)
                            except ValueError:
                                # If parsing fails, just use goal * 100
                                sdg_codes.append(goal * 100)
                    else:
                        # No targets specified, only goal level: e.g., 300
                        sdg_codes.append(goal * 100)

        # Calculate average confidence (for reporting purposes)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        return sdg_codes, avg_confidence
    except Exception as e:
        print(f"Parse error: {e}")
        if response:
            print(f"Response was: {response.output_text[:200] if 'response' in locals() else 'No response'}")
        return [], 0


# Parse command line arguments
parser = argparse.ArgumentParser(description='Classify SDG targets using OpenAI Responses API')
parser.add_argument('--input', '-i', type=str, default='CRS_somalia.csv',
                    help='Input CSV file path (default: CRS_somalia.csv)')
parser.add_argument('--output', '-o', type=str, default='sdg_results_openai.csv',
                    help='Output CSV file path (default: sdg_results_openai.csv)')
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output

print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")

# Load input data
df = pd.read_csv(INPUT_FILE)

# Check if output file exists and load existing results
if os.path.exists(OUTPUT_FILE):
    print(f"Found existing results file: {OUTPUT_FILE}")
    existing_results = pd.read_csv(OUTPUT_FILE)

    # Determine how many rows have been processed
    processed_count = len(existing_results)
    print(f"Already processed {processed_count} rows. Resuming from row {processed_count}...")

    # Start from where we left off
    start_idx = processed_count
else:
    print(f"No existing results found. Starting from beginning...")
    existing_results = None
    start_idx = 0

# Process remaining rows
for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df)):
    row = df.iloc[idx]

    # Classify the row
    sdg_codes, conf = classify_sdg(row)

    # Create result row combining input data and predictions
    result_row = row.to_dict()
    result_row["sdg_codes_predicted"] = sdg_codes
    result_row["confidence"] = conf

    # Append to output file
    result_df = pd.DataFrame([result_row])
    if idx == 0:
        # First row: create new file with header
        result_df.to_csv(OUTPUT_FILE, mode='w', index=False, header=True)
    else:
        # Subsequent rows: append without header
        result_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

print(f"\nProcessing complete! Results saved to {OUTPUT_FILE}")
