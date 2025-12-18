#!/bin/bash

# Find gpu4pyscf package path
GPU4PYSCF_PATH=$(python3 -c "import gpu4pyscf; print(gpu4pyscf.__path__[0])" 2>/dev/null)

if [ -z "$GPU4PYSCF_PATH" ]; then
    echo "Error: gpu4pyscf package not found"
    exit 1
fi

TARGET_FILE="$GPU4PYSCF_PATH/tools/ase_interface.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: ase_interface.py not found at $TARGET_FILE"
    exit 1
fi

echo "Found gpu4pyscf at: $GPU4PYSCF_PATH"
echo "Patching: $TARGET_FILE"

# Create patch file
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

# Apply patch
cd "$GPU4PYSCF_PATH/tools"
patch -p1 --forward < "$PATCH_FILE"
RESULT=$?

# Cleanup
rm -f "$PATCH_FILE"

if [ $RESULT -eq 0 ]; then
    echo "Patch applied successfully!"
elif [ $RESULT -eq 1 ]; then
    echo "Patch already applied or partially applied"
else
    echo "Error applying patch"
    exit 1
fi
