# SDG Classifier

Classification of development projects to UN Sustainable Development Goals (SDGs) using OpenAI API.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Scripts

### classify_sdg_openai.py
Single-request classification using OpenAI Responses API.

**Usage:**
```bash
python classify_sdg_openai.py --input CRS_somalia.csv --output sdg_results_openai.csv
```

**Features:**
- Processes projects one by one
- Supports resume from interruption
- Classifies to SDG target level (e.g., 301 for Goal 3, Target 1)
- Filters results by confidence threshold (0.8 by default)

### classify_sdg_openai_batch.py
Batch processing using OpenAI Batch API with 50% cost savings.

**Usage:**
```bash
# Create and submit batch job
python classify_sdg_openai_batch.py --create-batch --input CRS_somalia.csv --output sdg_results_batch.csv

# Check status of existing batch
python classify_sdg_openai_batch.py --check-status BATCH_JOB_ID

# Download results from completed batch
python classify_sdg_openai_batch.py --download-results BATCH_JOB_ID --input CRS_somalia.csv --output sdg_results_batch.csv
```

**Features:**
- 50% cost reduction compared to standard API
- Processes large datasets efficiently
- Displays token usage and cost breakdown
- Automatic wait and download when creating batch

### evaluate_openai_classifier.py
Evaluates classifier accuracy against ground truth data.

**Usage:**
```bash
python evaluate_openai_classifier.py --samples 100 --input ground_truth.csv --output evaluation_results.csv
```

**Features:**
- Tests classifier on labeled data
- Calculates exact match and partial match accuracy
- Adjustable confidence threshold
- Detailed per-sample results

## Input Format

CSV files should contain columns:
- `project_title` / `ProjectTitle`
- `short_description` / `ShortDescription`
- `long_description` / `LongDescription`
- `donor_name` / `DonorName`
- `agency_name` / `AgencyName`
- `purpose_name` / `PurposeName`
- `sector_name` / `SectorName`

## Output Format

Results include:
- All input columns
- `sdg_codes_predicted`: List of SDG codes (e.g., [301, 302] for Goal 3, Targets 1 and 2)
- `confidence`: Average confidence score (0-1)

## SDG Code Format

- Target level: `301` = Goal 3, Target 1
- Goal level: `300` = Goal 3 (no specific target)
- Empty list: Does not connect to any SDG
