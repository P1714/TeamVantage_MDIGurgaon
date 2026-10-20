.PHONY: all data test dashboard docs
all:            ## full rebuild: data -> twin -> models -> figures -> console
	python -m madhutwin.pipeline --regen
	python scripts/build_dashboard.py
test:
	python -m pytest -q
dashboard:
	python scripts/build_dashboard.py
docs:           ## architecture PDF, deck, README (needs playwright + node pptxgenjs)
	python scripts/render_docs.py
	node docs/src/build_deck.js
	python scripts/build_readme.py
