# Ideal Contract Templates Folder

Place your ideal contract PDF templates in this folder to process them as benchmarks for the Guardian Score system.

## File Naming Convention

Use the format: `{category}_{description}.pdf`

### Valid Categories:
- `rental` - Leave & License Agreements, Rental contracts
- `employment` - Employment agreements, service contracts  
- `nda` - Non-disclosure agreements
- `service_agreement` - Professional service contracts
- `partnership` - Partnership deeds
- `purchase` - Sale/purchase agreements  
- `lease` - Commercial lease agreements
- `llp` - Limited Liability Partnership agreements
- `property_sale` - Property sale agreements
- `loan_agreement` - Loan contracts

### Example Filenames:
- `rental_mumbai_leave_license.pdf`
- `employment_standard_indian.pdf`
- `nda_mutual_template.pdf`
- `partnership_deed_sample.pdf`

## How to Process Templates

1. Add your PDF files to this folder with the correct naming format
2. Call the API endpoint: `POST /api/v1/ragsys/ideal-contracts/process-folder-templates`
3. The system will automatically:
   - Extract text from PDFs
   - Classify them by category
   - Create embeddings for comparison
   - Store them as ideal templates

## Usage

Once processed, these templates will be used to:
- Compare uploaded user contracts
- Identify missing protections
- Detect unfair clauses
- Generate Guardian Scores (0-100)
- Protect users from exploitation

Add your best, most protective contract templates here!