#!/bin/bash

# Find gpu4pyscf package path
GPU4PYSCF_PATH=$(python3 -c "import gpu4pyscf; print(gpu4pyscf.__path__[0])" 2>/dev/null)

if [ -z "$GPU4PYSCF_PATH" ]; then
    echo "Error: gpu4pyscf package not found"
    exit 1
fi

echo "Found gpu4pyscf at: $GPU4PYSCF_PATH"

#############################################
# Patch 1: ase_interface.py - jqc support
#############################################
TARGET_FILE="$GPU4PYSCF_PATH/tools/ase_interface.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: ase_interface.py not found at $TARGET_FILE"
    exit 1
fi

echo "Patching: $TARGET_FILE"

PATCH_FILE=$(mktemp)
cat > "$PATCH_FILE" << 'EOF'
--- a/ase_interface.py
+++ b/ase_interface.py
@@ -20,6 +20,14 @@
           """)
     raise RuntimeError("ASE is not found")

+# Optional jqc support for JIT-compiled PySCF kernels
+try:
+    import jqc.pyscf as jqc_pyscf
+    HAS_JQC = True
+except ImportError:
+    jqc_pyscf = None
+    HAS_JQC = False
+
 import numpy as np
 from ase.units import Debye
 from pyscf import lib
@@ -48,7 +56,7 @@
     default_parameters = {}

     def __init__(self, restart=None, label='PySCF', atoms=None, directory='.',
-                 method=None, **kwargs):
+                 method=None, use_jqc=False, **kwargs):
         """Construct PySCF-calculator object.

         Parameters
@@ -58,6 +66,11 @@
             Default is 'PySCF'.

         method: A PySCF method class
+
+        use_jqc: bool
+            If True and jqc package is available, apply JIT-compiled PySCF
+            kernels using jqc.pyscf.apply(). This can provide performance
+            improvements for certain calculations. Default is False.
         """
         Calculator.__init__(self, restart, label=label, atoms=atoms,
                             directory=directory, **kwargs)
@@ -65,6 +78,15 @@
         if not isinstance(method, lib.StreamObject):
             raise RuntimeError(f'{method} must be an instance of a PySCF method')

+        # Apply jqc JIT-compilation if requested and available
+        self.use_jqc = use_jqc
+        if use_jqc:
+            if HAS_JQC:
+                method = jqc_pyscf.apply(method)
+            else:
+                import warnings
+                warnings.warn('jqc package not found. Install it via: pip install jqc')
+
         self.method = method
         self.pbc = hasattr(method, 'cell')
         if self.pbc:
EOF

cd "$GPU4PYSCF_PATH/tools"
patch -p1 --forward < "$PATCH_FILE" 2>/dev/null
RESULT=$?
rm -f "$PATCH_FILE"

if [ $RESULT -eq 0 ]; then
    echo "  -> ase_interface.py patched successfully!"
elif [ $RESULT -eq 1 ]; then
    echo "  -> ase_interface.py already patched"
else
    echo "  -> Error patching ase_interface.py"
fi

#############################################
# Patch 2: method_config.py - method support
#############################################
TARGET_FILE="$GPU4PYSCF_PATH/tools/method_config.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: method_config.py not found at $TARGET_FILE"
    exit 1
fi

echo "Patching: $TARGET_FILE"

PATCH_FILE=$(mktemp)
cat > "$PATCH_FILE" << 'EOF'
--- a/method_config.py
+++ b/method_config.py
@@ -35,6 +35,7 @@
         'spin': None,
         'xc': 'b3lyp',
         'disp': None,
+        'method': None,
         'grids': {'atom_grid': (99,590)},
         'nlcgrids': {'atom_grid': (50,194)},
         'basis': 'def2-tzvpp',
@@ -67,14 +68,37 @@
     if xc == 'LDA':
         xc = 'LDA,VWN5'

-    if xc.lower() == 'hf':
-        mf = scf.HF(mol)
+    # Method mapping for explicit method specification
+    method_map = {
+        # HF methods
+        'hf': scf.HF, 'rhf': scf.RHF, 'uhf': scf.UHF, 'rohf': scf.ROHF,
+        'ghf': scf.GHF,
+        # DFT methods
+        'ks': dft.KS, 'rks': dft.RKS, 'uks': dft.UKS, 'roks': dft.ROKS,
+        'gks': dft.GKS,
+    }
+
+    method_type = config.get('method')
+    if method_type is not None:
+        method_key = method_type.lower()
+        if method_key not in method_map:
+            raise ValueError(f"Unknown method: {method_type}. "
+                           f"Supported: {list(method_map.keys())}")
+        method_cls = method_map[method_key]
+        if method_key in ('hf', 'rhf', 'uhf', 'rohf', 'ghf'):
+            mf = method_cls(mol)
+        else:
+            mf = method_cls(mol, xc=xc)
+    elif xc.lower() == 'hf':
+        mf = scf.HF(mol)
     else:
         mf = dft.KS(mol, xc=xc)
-        grids = config['grids']
-        nlcgrids = config['nlcgrids']
-        if 'atom_grid' in grids: mf.grids.atom_grid = grids['atom_grid']
-        if 'level' in grids:     mf.grids.level     = grids['level']
+
+    # Apply grid settings for DFT methods
+    if hasattr(mf, 'grids'):
+        grids = config.get('grids', {})
+        nlcgrids = config.get('nlcgrids', {})
+        if 'atom_grid' in grids: mf.grids.atom_grid = grids['atom_grid']
+        if 'level' in grids:     mf.grids.level     = grids['level']
         if mf._numint.libxc.is_nlc(mf.xc):
             if 'atom_grid' in nlcgrids: mf.nlcgrids.atom_grid = nlcgrids['atom_grid']
             if 'level' in nlcgrids:     mf.nlcgrids.level     = nlcgrids['level']
EOF

cd "$GPU4PYSCF_PATH/tools"
patch -p1 --forward < "$PATCH_FILE" 2>/dev/null
RESULT=$?
rm -f "$PATCH_FILE"

if [ $RESULT -eq 0 ]; then
    echo "  -> method_config.py patched successfully!"
elif [ $RESULT -eq 1 ]; then
    echo "  -> method_config.py already patched"
else
    echo "  -> Error patching method_config.py"
fi

echo ""
echo "Done! Usage example:"
echo "  config = get_default_config()"
echo "  config['method'] = 'uks'  # or 'rks', 'uhf', 'rhf', etc."
echo "  mf = method_from_config(config)"
echo "  atoms.calc = PySCF(method=mf, use_jqc=True)"
