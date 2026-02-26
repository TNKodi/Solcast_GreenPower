# Solcast_GreenPower

## AWS SAM Lambda Structure

This repository now includes SAM-ready deployment files for AWS Lambda:

- `template.yaml` - SAM template
- `src/lambda_handler.py` - Lambda entrypoint using Mangum
- `app/` - FastAPI application code
- `Scripts/power_predicition/` - Forecast logic

## Deploy with SAM

```powershell
sam build
sam deploy --guided
```

## Run Locally with SAM

```powershell
sam build
sam local start-api
```