import base64
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from data_generation import create_data_from_params  # noqa: E402
from functions import (  # noqa: E402
	DashboardFilesMissing,
	api_error,
	api_success,
	get_all_dataset_files,
	get_dataset_dashboard_files,
	get_dataset_names,
)

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
async def list_datasets():
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
async def dataset_dashboard(dataset_name: str):
	try:
		data = get_dataset_dashboard_files(dataset_name, datasets_dir=DATASETS_DIR)

		return api_success(message=f"Success: Dashboard files returned for dataset '{dataset_name}'", data=data, status_code=200)

	except DashboardFilesMissing as e:
		return api_error(
			message=f"Missing required files for dataset '{e.dataset_name}': {', '.join(e.missing)}", status_code=404, code='DASHBOARD_FILES_MISSING'
		)

	except UnicodeDecodeError:
		return api_error(message='Error: Could not decode one of the CSV files (encoding issue)', status_code=500, code='CSV_DECODE_FAILED')

	except Exception as e:
		print(f'Error: Dashboard fetch failed: {str(e)}')
		return api_error(message='Unexpected server error while building dashboard response', status_code=500, code='DASHBOARD_FAILED')


@app.get('/api/dataset/{dataset_name}/download')
async def download_dataset(dataset_name: str):
	"""
	Download ALL dataset files as a single zip.
	"""
	try:
		files = get_all_dataset_files(dataset_name, datasets_dir=DATASETS_DIR)

		# Build zip in-memory
		buf = io.BytesIO()
		with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
			# Text files
			zf.writestr(f'{dataset_name}.truth_genotypes.csv', files['truth_genotypes_csv'])
			zf.writestr(f'{dataset_name}.observed_genotypes.csv', files['observed_genotypes_csv'])
			zf.writestr(f'{dataset_name}.pedigree.csv', files['pedigree_csv'])
			zf.writestr(f'{dataset_name}.run_metadata.json', files['run_metadata_json'])

			# Binary trees (base64 -> bytes)
			trees_bytes = base64.b64decode(files['trees_base64'])
			zf.writestr(f'{dataset_name}.trees', trees_bytes)

		buf.seek(0)

		return StreamingResponse(buf, media_type='application/zip', headers={'Content-Disposition': f'attachment; filename="{dataset_name}.zip"'})

	except DashboardFilesMissing as e:
		return api_error(
			message=f"Missing required files for dataset '{e.dataset_name}': {', '.join(e.missing)}", status_code=404, code='DATASET_FILES_MISSING'
		)

	except Exception as e:
		print(f'Error: Dataset download failed: {str(e)}')
		return api_error(message='Unexpected server error while building dataset zip', status_code=500, code='DATASET_ZIP_FAILED')
