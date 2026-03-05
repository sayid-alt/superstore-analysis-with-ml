deactivate

if [[ -d ".venv" ]]; then
	echo "Virtual environment already exists. Delete it..." 
	rm -rf .venv;
fi

echo "Creating virtual environment"
uv sync

echo "Activating virtual environment..."
source ./.venv/bin/activate
