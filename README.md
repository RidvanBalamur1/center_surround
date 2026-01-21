project_name/
├── project_name/ # Your importable source code
│ ├── **init**.py # Makes it a Python package
│ ├── data/ # Data loading & preprocessing
│ ├── models/ # Model class definitions
│ ├── training/ # Training loops, callbacks
│ └── utils/ # Helper functions
├── scripts/ # CLI entry points (train.py, etc.)
├── configs/ # YAML/JSON config files
├── tests/ # Unit tests
├── data/ # Raw/processed data (gitignored)
├── outputs/ # Checkpoints, logs, results (gitignored)
├── notebooks/ # Jupyter notebooks for exploration
├── requirements.txt # pip dependencies
├── setup.py # Makes project pip-installable
├── README.md # Project documentation
└── .gitignore # Files git should ignore
