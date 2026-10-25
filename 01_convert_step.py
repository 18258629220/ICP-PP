"""STEP -> binary STL with absolute OCCT meshing deflection (mm)."""
import logging
import re
import time
from pathlib import Path
import cadquery as cq
from pipeline_common import ROOT, config, dump_json, log_setup, sha256

def main():
    cfg=config(); out=ROOT/cfg['output_dir']; out.mkdir(exist_ok=True); log_setup(out)
    start=time.time(); step=ROOT/cfg['step_path']
    text=step.read_text(errors='strict')
    if not re.search(r'SI_UNIT\(\.MILLI\.,\.METRE\.\)',text):
        raise ValueError('Expected explicit mm STEP length unit; inspect before proceeding')
    uncertainties=re.findall(r'UNCERTAINTY_MEASURE_WITH_UNIT\(LENGTH_MEASURE\(([^)]+)',text)
    logging.info('Importing STEP; mm confirmed; declared model uncertainty=%s',uncertainties)
    del text
    wp=cq.importers.importStep(str(step))
    shapes=wp.vals()
    shape=shapes[0] if len(shapes)==1 else cq.Compound.makeCompound(shapes)
    bbox=shape.BoundingBox()
    logging.info('STEP imported: %d roots; bbox=%s',len(shapes),[bbox.xlen,bbox.ylen,bbox.zlen])
    meta={'units':'mm','sha256':sha256(step),'cadquery_version':cq.__version__,
          'step_declared_uncertainty_mm':[float(x) for x in uncertainties],
          'imported_roots':len(shapes),'cad_valid':shape.isValid(),'solids':len(shape.Solids()),'shells':len(shape.Shells()),
          'faces':len(shape.Faces()),'cad_volume_mm3':shape.Volume(),
          'volume_is_material_volume':len(shape.Solids())>0,
          'bbox_min_mm':[bbox.xmin,bbox.ymin,bbox.zmin],
          'bbox_max_mm':[bbox.xmax,bbox.ymax,bbox.zmax],
          'linear_deflection_mm':cfg['mesh_linear_deflection_mm'],
          'angular_deflection_rad':cfg['mesh_angular_deflection_rad'], 'relative_deflection':False}
    # CadQuery exportStl defaults to relative=True; explicitly use absolute mm here.
    ok=shape.exportStl(str(out/'cad_mesh.stl'),tolerance=cfg['mesh_linear_deflection_mm'],
        angularTolerance=cfg['mesh_angular_deflection_rad'],ascii=False,relative=False,parallel=False)
    if not ok: raise RuntimeError('STL export failed')
    meta['elapsed_seconds']=time.time()-start
    dump_json(out/'cad_metadata.json',meta); logging.info('CAD export complete: %s',meta)
if __name__=='__main__': main()
