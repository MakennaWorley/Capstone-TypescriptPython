import base64
import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from api import api_error, api_success  # noqa: E402
from data_generation import create_data_from_params  # noqa: E402
from functions import get_dataset_names  # noqa: E402

app = FastAPI()

origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:8501').split(',')
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'datasets'


# debugging
@app.on_event('startup')
async def show_routes():
	print('=== ROUTES ===')
	for r in app.routes:
		try:
			print(f'{getattr(r, "methods", "")} {r.path}')
		except Exception:
			pass
	print('==============')


# health check
@app.get('/api/hello')
def hello():
	return {'message': 'Hello from FastAPI'}


@app.post('/api/create/data')
async def create_dataset(request: Request):
	try:
		body = await request.json()
	except Exception:
		return api_error(message='Error: Request body must be valid JSON', status_code=400, code='INVALID_JSON')

	try:
		simulation_params = body.get('params', body)

		if not isinstance(simulation_params, dict) or len(simulation_params) == 0:
			return api_error(message='Error: Missing or invalid simulation params', status_code=400, code='INVALID_PARAMS')

		os.makedirs(DATASETS_DIR, exist_ok=True)
		simulation_params['output_dir'] = str(DATASETS_DIR)

		res = create_data_from_params(simulation_params)

		return api_success(
			message='Success: Data generated successfully',
			data={
				'dataset_name': simulation_params.get('name'),
				'output_dir': str(DATASETS_DIR),
				'result': {'config_used': res['config'], 'file_paths': res['outputs']},
			},
			status_code=200,
		)

	except Exception as e:
		print(f'Error during data generation: {str(e)}')
		return api_error(message='Unexpected server error while generating data', status_code=500, code='DATASET_CREATE_FAILED')


# Get Dataset
@app.get('/api/datasets/list', response_model=List[str])
async def list_datasets(request: Request):
	try:
		dataset_names = get_dataset_names()

		return api_success(
			message='Success: Datasets retrieved successfully', data={'datasets': dataset_names, 'count': len(dataset_names)}, status_code=200
		)

	except UnicodeDecodeError:
		return api_error(message='Error: Could not find dataset list', status_code=500, code='DATASET_FILE_MISSING')

	except Exception as e:
		print(f'Error: Could not read dataset list: {str(e)}')
		return api_error(message='Unexpected server error while reading datasets', status_code=500, code='DATASET_LIST_FAILED')


@app.get('/api/dataset/{dataset_name}/dashboard')
async def dataset_dashboard(dataset_name: str, request: Request):
	try:
		trees_path = DATASETS_DIR / f'{dataset_name}.trees'
		truth_path = DATASETS_DIR / f'{dataset_name}.truth_genotypes.csv'
		observed_path = DATASETS_DIR / f'{dataset_name}.observed_genotypes.csv'

		print('DATASETS_DIR =', DATASETS_DIR.resolve())
		print('dataset_name =', dataset_name)
		print('trees_path   =', trees_path.resolve(), 'exists?', trees_path.exists())
		print('truth_path   =', truth_path.resolve(), 'exists?', truth_path.exists())
		print('observed_path=', observed_path.resolve(), 'exists?', observed_path.exists())

		missing = []
		if not trees_path.exists():
			missing.append(trees_path.name)
		if not truth_path.exists():
			missing.append(truth_path.name)
		if not observed_path.exists():
			missing.append(observed_path.name)

		if missing:
			return api_error(
				message=f"Missing required files for dataset '{dataset_name}': {', '.join(missing)}", status_code=404, code='DASHBOARD_FILES_MISSING'
			)

		# Read CSVs as text
		truth_csv = truth_path.read_text(encoding='utf-8')
		observed_csv = observed_path.read_text(encoding='utf-8')

		# Read trees as bytes -> base64
		trees_bytes = trees_path.read_bytes()
		trees_b64 = base64.b64encode(trees_bytes).decode('ascii')

		return api_success(
			message=f"Success: Dashboard files returned for dataset '{dataset_name}'",
			data={
				'dataset': dataset_name,
				'observed_genotypes_csv': observed_csv,
				'truth_genotypes_csv': truth_csv,
				'trees_name': trees_path.name,
				'trees_base64': trees_b64,
				'trees_byte_length': len(trees_bytes),
			},
			status_code=200,
		)

	except UnicodeDecodeError:
		return api_error(message='Error: Could not decode one of the CSV files (encoding issue)', status_code=500, code='CSV_DECODE_FAILED')

	except ValueError as ve:
		return api_error(message=str(ve), status_code=400, code='INVALID_DATASET_NAME')

	except Exception as e:
		print(f'Error: Dashboard fetch failed: {str(e)}')
		return api_error(message='Unexpected server error while building dashboard response', status_code=500, code='DASHBOARD_FAILED')
