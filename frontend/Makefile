.PHONY: help dev init build

help:
	@echo "Available commands:"
	@echo "  make init   - Initializes the project runtime"
	@echo "  make build  - Builds the project"
	@echo "  make dev    - Starts the frontend development server (Vite)"

dev:
	@echo "Starting frontend development server..."
	@npm run dev

init:
	@echo "Preparing frontend npm packages"
	@npm install
	@npm audit fix

build:
	@echo "Building site for production"
	@npm run build