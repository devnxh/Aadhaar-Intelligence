"""
Script to execute all analysis notebooks in sequence
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
from pathlib import Path

def run_notebook(notebook_path):
    """Execute a Jupyter notebook"""
    print(f"\n{'='*70}")
    print(f"Executing: {notebook_path.name}")
    print('='*70)
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})
        
        # Save executed notebook
        output_path = notebook_path.parent / f"{notebook_path.stem}_executed.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✅ Successfully executed: {notebook_path.name}")
        print(f"   Output saved to: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error executing {notebook_path.name}: {str(e)}")
        print(f"   Continuing with next notebook...")
        return False

if __name__ == "__main__":
    notebooks_dir = Path("notebooks")
    
    # List of notebooks to execute in order
    notebooks = [
        "01_data_cleaning.ipynb",
        "02_eda.ipynb",
        "03_indicators.ipynb",
        "04_anomaly_detection.ipynb",
        "05_forecasting.ipynb"
    ]
    
    results = {}
    
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            success = run_notebook(nb_path)
            results[nb_name] = success
        else:
            print(f"⚠️ Notebook not found: {nb_name}")
            results[nb_name] = False
    
    # Summary
    print(f"\n\n{'='*70}")
    print("EXECUTION SUMMARY")
    print('='*70)
    
    for nb_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {nb_name}")
    
    print('='*70)
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nCompleted: {successful}/{total} notebooks")
    
    if successful == total:
        print("\n🎉 All notebooks executed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - successful} notebook(s) had issues")
        sys.exit(1)
