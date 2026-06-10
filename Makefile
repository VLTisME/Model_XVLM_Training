.PHONY: install test lint overfit train eval clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q --cov=star --cov-report=term-missing

lint:
	ruff check src tests scripts
	black --check src tests scripts

# NOTE: the data manifest + images are delivered by the DATA TEAM (docs/02_DATA_CONTRACT.md).
overfit:
	python scripts/train.py --config configs/star_v3_100k.yaml --overfit-one-batch

train:
	python scripts/train.py --config configs/star_v3_100k.yaml

eval:
	python scripts/evaluate.py --config configs/star_v3_100k.yaml --ckpt checkpoints/best.pth

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov outputs/tmp
