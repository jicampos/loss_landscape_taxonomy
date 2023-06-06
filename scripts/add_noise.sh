python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type bernoulli --noise-magnitude 1.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type bernoulli --noise-magnitude 2.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type bernoulli --noise-magnitude 4.0 --save-path data/JT

python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type gaussian --noise-magnitude 1.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type gaussian --noise-magnitude 2.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type gaussian --noise-magnitude 4.0 --save-path data/JT

python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type uniform --noise-magnitude 1.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type uniform --noise-magnitude 2.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type uniform --noise-magnitude 4.0 --save-path data/JT

python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type positive_uniform --noise-magnitude 1.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type positive_uniform --noise-magnitude 2.0 --save-path data/JT
python scripts/jet_dataset.py --config scripts/config.yml --noise --noise-type positive_uniform --noise-magnitude 4.0 --save-path data/JT
