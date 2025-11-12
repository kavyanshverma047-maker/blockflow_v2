# Makefile
# Blockflow Exchange - Development Commands

.PHONY: help install run test migrate docker-up docker-down clean demo

help:
	@echo "Blockflow Exchange - Available Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make run          - Run development server"
	@echo "  make test         - Run tests"
	@echo "  make migrate      - Run database migrations"
	@echo "  make docker-up    - Start Docker containers"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make demo         - Run trading demo"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt

run:
	uvicorn main:app --reload --port 8000

test:
	pytest tests/ -v --cov=. --cov-report=html

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(message)"

docker-up:
	docker-compose up --build -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Services ready! API: http://localhost:8000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

demo:
	python demo_trading.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf logs/*.log