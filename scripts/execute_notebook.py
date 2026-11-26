"""Execute the report notebook in the repository's local registered kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
p=root/'Final_Project.ipynb'
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=180,kernel_name='isc-project',resources={'metadata':{'path':str(root)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,p)
print('Executed and validated Final_Project.ipynb')
