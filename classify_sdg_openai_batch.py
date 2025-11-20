from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json
import re
import os
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_batch_request(row, custom_id):
    """
    Create a single batch request for a project row.
    """
    title = row.get("ProjectTitle", "")
    short_desc = row.get("ShortDescription", "")
    long_desc = row.get("LongDescription", "")
    donor = row.get("DonorName", "")
    agency = row.get("AgencyName", "")
    purpose = row.get("PurposeName", "")
    sector = row.get("SectorName", "")

    # title = row.get("project_title", "")
    # short_desc = row.get("short_description", "")
    # long_desc = row.get("long_description", "")
    # donor = row.get("donor_name", "")
    # agency = row.get("agency_name", "")
    # purpose = row.get("purpose_name", "")
    # sector = row.get("sector_name", "")

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
       "logistics support", "program management"), then classify it as **DNC (Does Not Connect)**.
        - DNC means the project does not have a clear, direct link to any SDG target.
       - Avoid forcing a classification when there is no clear substantive focus.
    5. Return the results in a valid JSON object.
    6. If the project clearly does not relate to any SDG target, but it can be clearly related at the goal level, return a goal.
    7. If the project clearly does not relate to any SDG target nor a goal, return an empty list.

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
    **Example response:**
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

    ---
    **Project Information:**
    {project_text}

    Use only the official UN SDG target definitions when making your decision.
    """

    combined_prompt = f"""You are a highly accurate SDG classification model that maps projects to official UN SDG targets.

{user_prompt}"""

    # Create batch request in the required format
    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4o-mini",
            "input": combined_prompt
        }
    }

    return batch_request


def create_batch_input_file(df, start_idx=0, end_idx=None, batch_file="batch_input.jsonl"):
    """
    Create a JSONL file with all batch requests.
    """
    if end_idx is None:
        end_idx = len(df)

    with open(batch_file, 'w') as f:
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            custom_id = f"request-{idx}"
            batch_request = create_batch_request(row, custom_id)
            f.write(json.dumps(batch_request) + '\n')

    print(f"Created batch input file: {batch_file}")
    print(f"Total requests: {end_idx - start_idx}")
    return batch_file


def upload_batch_file(batch_file):
    """
    Upload the batch file to OpenAI.
    """
    print(f"Uploading batch file: {batch_file}")
    with open(batch_file, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    print(f"File uploaded successfully. File ID: {batch_input_file.id}")
    return batch_input_file.id


def create_batch_job(batch_input_file_id):
    """
    Create a batch processing job.
    """
    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": "SDG classification batch job"
        }
    )
    print(f"Batch job created. Job ID: {batch_job.id}")
    print(f"Status: {batch_job.status}")
    return batch_job.id


def check_batch_status(batch_job_id):
    """
    Check the status of a batch job.
    """
    batch_job = client.batches.retrieve(batch_job_id)
    return batch_job


def wait_for_batch_completion(batch_job_id, check_interval=60):
    """
    Wait for the batch job to complete, checking every check_interval seconds.
    """
    print(f"Waiting for batch job to complete. Checking every {check_interval} seconds...")
    print("This may take a while. You can also check status later using --check-status option.")

    while True:
        batch_job = check_batch_status(batch_job_id)
        status = batch_job.status

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")

        if status == "completed":
            print("Batch job completed successfully!")
            return batch_job
        elif status == "failed":
            print(f"Batch job failed. Error: {batch_job.errors}")
            return None
        elif status == "expired":
            print("Batch job expired.")
            return None
        elif status == "cancelled":
            print("Batch job was cancelled.")
            return None

        # Show progress if available
        if hasattr(batch_job, 'request_counts'):
            counts = batch_job.request_counts
            total = counts.total if hasattr(counts, 'total') else 0
            completed = counts.completed if hasattr(counts, 'completed') else 0
            failed = counts.failed if hasattr(counts, 'failed') else 0

            if total > 0:
                progress = (completed / total) * 100
                print(f"  Progress: {completed}/{total} ({progress:.1f}%) | Failed: {failed}")

        time.sleep(check_interval)


def download_batch_results(batch_job, output_file="batch_output.jsonl"):
    """
    Download the results of a completed batch job.
    """
    if not batch_job.output_file_id:
        print("No output file available.")
        return None

    print(f"Downloading results from file ID: {batch_job.output_file_id}")
    file_response = client.files.content(batch_job.output_file_id)

    with open(output_file, 'wb') as f:
        f.write(file_response.content)

    print(f"Results downloaded to: {output_file}")
    return output_file


def parse_batch_response(response_text):
    """
    Parse a single batch response and extract SDG codes.
    """
    try:
        # Remove markdown code blocks if present (```json ... ``` or ``` ... ```)
        content_fixed = response_text.strip()
        if content_fixed.startswith('```'):
            # Remove opening ```json or ```
            content_fixed = re.sub(r'^```(?:json)?\s*\n?', '', content_fixed)
            # Remove closing ```
            content_fixed = re.sub(r'\n?```\s*$', '', content_fixed)

        # Extract JSON safely - handle cases where JSON might be malformed
        content_fixed = content_fixed.replace('"goal": DNC', '"goal": "DNC"')

        json_match = re.search(r"\{.*\}", content_fixed, re.S)
        if not json_match:
            return [], 0

        try:
            result = json.loads(json_match.group(0))
        except json.JSONDecodeError:
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
        return [], 0


def process_batch_results(batch_output_file, input_df, output_csv):
    """
    Process the batch results and create the final CSV output.
    """
    print(f"Processing batch results from: {batch_output_file}")

    # Read batch results
    results = {}
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    # OpenAI Batch API pricing for gpt-4o-mini (50% discount)
    # https://openai.com/api/pricing/
    PRICE_PER_1M_INPUT_TOKENS = 0.075  # USD
    PRICE_PER_1M_OUTPUT_TOKENS = 0.300  # USD

    with open(batch_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']

            # Extract row index from custom_id (format: "request-{idx}")
            idx = int(custom_id.split('-')[1])

            if result.get('error'):
                print(f"Error in request {custom_id}: {result['error']}")
                results[idx] = ([], 0)
                error_count += 1
            else:
                # Extract the response text from the correct path
                try:
                    response_body = result['response']['body']
                    output = response_body.get('output', [])

                    if output and len(output) > 0:
                        content = output[0].get('content', [])
                        if content and len(content) > 0:
                            output_text = content[0].get('text', '')
                        else:
                            output_text = ''
                    else:
                        output_text = ''

                    # Parse the response
                    sdg_codes, confidence = parse_batch_response(output_text)
                    results[idx] = (sdg_codes, confidence)

                    # Extract token usage
                    usage = response_body.get('usage', {})
                    if usage:
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        tokens = usage.get('total_tokens', 0)

                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        total_tokens += tokens

                except (KeyError, IndexError, TypeError) as e:
                    print(f"Error parsing response for {custom_id}: {e}")
                    results[idx] = ([], 0)
                    error_count += 1

    # Create output dataframe
    output_rows = []
    for idx in range(len(input_df)):
        row = input_df.iloc[idx].to_dict()

        if idx in results:
            sdg_codes, confidence = results[idx]
            row["sdg_codes_predicted"] = sdg_codes
            row["confidence"] = confidence
        else:
            row["sdg_codes_predicted"] = []
            row["confidence"] = 0

        output_rows.append(row)

    # Save to CSV
    result_df = pd.DataFrame(output_rows)
    result_df.to_csv(output_csv, index=False)

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    output_cost = (total_output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    total_cost = input_cost + output_cost

    # Calculate what it would have cost with standard API (for comparison)
    standard_input_price = 0.150  # USD per 1M tokens
    standard_output_price = 0.600  # USD per 1M tokens
    standard_cost = ((total_input_tokens / 1_000_000) * standard_input_price +
                     (total_output_tokens / 1_000_000) * standard_output_price)
    savings = standard_cost - total_cost
    savings_percentage = (savings / standard_cost * 100) if standard_cost > 0 else 0

    # Print summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nRequests:")
    print(f"  Total requests:  {len(input_df)}")
    print(f"  Successful:      {len(results) - error_count}")
    print(f"  Errors:          {error_count}")

    print(f"\nToken Usage:")
    print(f"  Input tokens:    {total_input_tokens:,}")
    print(f"  Output tokens:   {total_output_tokens:,}")
    print(f"  Total tokens:    {total_tokens:,}")

    print(f"\nCost Breakdown (Batch API - 50% discount):")
    print(f"  Input cost:      ${input_cost:.4f}  ({total_input_tokens:,} tokens @ ${PRICE_PER_1M_INPUT_TOKENS}/1M)")
    print(f"  Output cost:     ${output_cost:.4f}  ({total_output_tokens:,} tokens @ ${PRICE_PER_1M_OUTPUT_TOKENS}/1M)")
    print(f"  Total cost:      ${total_cost:.4f}")

    print(f"\nCost Comparison:")
    print(f"  Standard API:    ${standard_cost:.4f}")
    print(f"  Batch API:       ${total_cost:.4f}")
    print(f"  Savings:         ${savings:.4f} ({savings_percentage:.1f}% discount)")

    print(f"\nOutput:")
    print(f"  Results saved to: {output_csv}")
    print(f"{'='*70}\n")

    return result_df


def main():
    parser = argparse.ArgumentParser(description='Classify SDG targets using OpenAI Batch API (50% cost reduction)')
    parser.add_argument('--input', '-i', type=str, default='CRS_somalia.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default='sdg_results_openai_batch.csv',
                        help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Number of rows to process (default: all rows)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting row index (default: 0)')
    parser.add_argument('--create-batch', action='store_true',
                        help='Create and submit a new batch job')
    parser.add_argument('--check-status', type=str, default=None,
                        help='Check status of an existing batch job by ID')
    parser.add_argument('--download-results', type=str, default=None,
                        help='Download results from a completed batch job by ID')
    parser.add_argument('--batch-file', type=str, default='batch_input.jsonl',
                        help='Batch input JSONL file path')
    parser.add_argument('--output-jsonl', type=str, default='batch_output.jsonl',
                        help='Batch output JSONL file path')
    parser.add_argument('--check-interval', type=int, default=60,
                        help='Interval in seconds to check batch status (default: 60)')

    args = parser.parse_args()

    # Check status of existing batch
    if args.check_status:
        batch_job = check_batch_status(args.check_status)
        print(f"Batch Job ID: {batch_job.id}")
        print(f"Status: {batch_job.status}")
        print(f"Created at: {batch_job.created_at}")

        if hasattr(batch_job, 'request_counts'):
            counts = batch_job.request_counts
            print(f"\nRequest counts:")
            print(f"  Total: {counts.total if hasattr(counts, 'total') else 'N/A'}")
            print(f"  Completed: {counts.completed if hasattr(counts, 'completed') else 'N/A'}")
            print(f"  Failed: {counts.failed if hasattr(counts, 'failed') else 'N/A'}")

        if batch_job.status == "completed":
            print(f"\nOutput file ID: {batch_job.output_file_id}")
            print(f"Use --download-results {batch_job.id} to download the results")

        return

    # Download results from completed batch
    if args.download_results:
        batch_job = check_batch_status(args.download_results)
        if batch_job.status != "completed":
            print(f"Batch job is not completed yet. Status: {batch_job.status}")
            return

        output_file = download_batch_results(batch_job, args.output_jsonl)
        if output_file:
            # Load input data and process results
            df = pd.read_csv(args.input)
            process_batch_results(output_file, df, args.output)
        return

    # Create and submit new batch
    if args.create_batch:
        # Load input data
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")

        # Determine batch size
        end_idx = args.start_idx + args.batch_size if args.batch_size else len(df)
        end_idx = min(end_idx, len(df))

        # Create batch input file
        batch_file = create_batch_input_file(df, args.start_idx, end_idx, args.batch_file)

        # Upload file
        file_id = upload_batch_file(batch_file)

        # Create batch job
        job_id = create_batch_job(file_id)

        print(f"\n{'='*60}")
        print(f"Batch job created successfully!")
        print(f"Job ID: {job_id}")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Wait for completion (may take several hours)")
        print(f"2. Check status: python classify_sdg_openai_batch.py --check-status {job_id}")
        print(f"3. Download results: python classify_sdg_openai_batch.py --download-results {job_id} -i {args.input} -o {args.output}")
        print(f"\nOr wait automatically:")

        # Wait for completion
        batch_job = wait_for_batch_completion(job_id, args.check_interval)

        if batch_job and batch_job.status == "completed":
            # Download and process results
            output_file = download_batch_results(batch_job, args.output_jsonl)
            if output_file:
                process_batch_results(output_file, df, args.output)

        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
