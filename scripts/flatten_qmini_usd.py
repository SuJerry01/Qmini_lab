# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Flatten a nested URDF-converted USD into the shipped-robot layout (like Isaac Lab's /h1).

The Isaac Sim 3.0 URDF importer emits the rigid-body link prims **nested** by kinematic depth
(``/YSXSZ/Geometry/base_link/hip_yaw_l/hip_roll_l/.../ankle_pitch_l``). Isaac Lab's ``ContactSensor``
matches bodies with the path regex ``/Robot/.*`` (one level), so it cannot find the deep feet and
``activate_contact_sensors`` only reaches ``base_link``. Shipped robots (e.g. ``/h1/<link>``) keep the
link prims **flat** — direct children of the articulation root — with the kinematic tree carried only by
the joints (``body0``/``body1``). This tool rewrites our USD into that flat layout.

What it does (and does NOT):
- Reparents every rigid-body link to be a direct child of the model root (``/YSXSZ/<link>``).
- Re-bakes each link's transform to its world pose (relative to the root) so geometry doesn't move.
- Repoints every joint's ``body0``/``body1`` to the new flat paths (joints are already flat under
  ``/YSXSZ/Physics`` — only their target paths change).
- Leaves untouched: mass/inertia/CoM, joint axis/limits/localPose/drive, meshes, materials, the
  articulation-root API.
- Self-verifies: asserts each link's world transform is unchanged after the rewrite.

Run (no GPU needed — pure USD editing):
    /workspace/isaaclab/isaaclab.sh -p scripts/flatten_qmini_usd.py --in <nested.usda> --out <flat.usda>
"""

from __future__ import annotations

import argparse

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

parser = argparse.ArgumentParser(description="Flatten a nested URDF-converted USD to the shipped flat layout.")
parser.add_argument("--in", dest="src", required=True, help="Input (nested) USD path.")
parser.add_argument("--out", dest="dst", required=True, help="Output (flat) USD path.")
parser.add_argument("--root", default="/YSXSZ", help="Model/articulation root prim path (default /YSXSZ).")
args = parser.parse_args()


def _close(a: Gf.Matrix4d, b: Gf.Matrix4d, tol: float = 1e-4) -> bool:
    return all(abs(a[i][j] - b[i][j]) <= tol for i in range(4) for j in range(4))


def main() -> None:
    src = Usd.Stage.Open(args.src)
    if not src:
        raise SystemExit(f"cannot open {args.src}")
    root = Sdf.Path(args.root)
    tc = Usd.TimeCode.Default()

    root_world = UsdGeom.Xformable(src.GetPrimAtPath(root)).ComputeLocalToWorldTransform(tc)
    root_world_inv = root_world.GetInverse()

    # gather every rigid-body link: (old_path, name, transform-relative-to-root, original world)
    links = []
    for p in src.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            w = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(tc)
            links.append((p.GetPath(), p.GetName(), w * root_world_inv, w))
    names = {n for _, n, _, _ in links}
    oldpath_to_new = {str(op): root.AppendChild(n) for op, n, _, _ in links}
    print(f"[flatten] {len(links)} rigid-body links found")

    # 1) flatten composition into one editable layer, then reparent each link to /root/<name>
    flat = src.Flatten()
    for old_path, name, _, _ in links:
        dst = root.AppendChild(name)
        if old_path != dst:
            Sdf.CopySpec(flat, old_path, flat, dst)

    st = Usd.Stage.Open(flat)

    # 2) at each flat link, drop child prims that are themselves rigid bodies (the nested next links);
    #    geometry children (visual/collision Xforms, meshes) have no RigidBodyAPI and are kept.
    for _, name, _, _ in links:
        dst = root.AppendChild(name)
        for child in list(st.GetPrimAtPath(dst).GetChildren()):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                st.RemovePrim(child.GetPath())

    # 3) remove the original nested subtree (everything was copied out to flat siblings)
    for top in {op for op in oldpath_to_new}:  # remove only the topmost original link under root
        pass
    # the topmost original link is the articulation root's body (e.g. /root/Geometry/base_link)
    for old_path, name, _, _ in links:
        # a link is "topmost-original" if no other link's old_path is its prefix
        if not any(old_path != o and str(old_path).startswith(str(o) + "/") for o, _, _, _ in links):
            if st.GetPrimAtPath(old_path):
                st.RemovePrim(old_path)

    # 4) re-bake transforms: each flat link's local transform = its world pose relative to root
    for _, name, rel, _ in links:
        xf = UsdGeom.Xformable(st.GetPrimAtPath(root.AppendChild(name)))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(rel)

    # 5) repoint every joint's body0/body1 to the new flat link paths
    for j in st.Traverse():
        if j.IsA(UsdPhysics.Joint):
            for rel_name in ("physics:body0", "physics:body1"):
                r = j.GetRelationship(rel_name)
                if not r:
                    continue
                new_targets = [oldpath_to_new.get(str(t), t) for t in r.GetTargets()]
                if new_targets:
                    r.SetTargets(new_targets)

    # 5b) FREE the base: deactivate the global fixed joint (world->base "root_joint", a URDF fix_base
    #     artifact). PhysX's fix_root_link=False can't reach it (it's a sibling of the articulation root,
    #     so find_global_fixed_joint_prim never sees it) and Newton's add_usd ignores jointEnabled — so we
    #     deactivate the prim outright (pruned from the stage → invisible to BOTH engines). A biped needs a
    #     free floating base to settle/walk. A world joint is a fixed joint with exactly one body target.
    for j in list(st.Traverse()):
        if j.IsA(UsdPhysics.Joint):
            t0 = j.GetRelationship("physics:body0").GetTargets() if j.GetRelationship("physics:body0") else []
            t1 = j.GetRelationship("physics:body1").GetTargets() if j.GetRelationship("physics:body1") else []
            if bool(t0) ^ bool(t1):  # exactly one side connected => joint to the world
                j.SetActive(False)
                print(f"[flatten] freed base: deactivated world fixed joint {j.GetPath()}")

    # 6) verify: every link's world transform must be unchanged
    bad = 0
    for _, name, _, orig_w in links:
        new_w = UsdGeom.Xformable(st.GetPrimAtPath(root.AppendChild(name))).ComputeLocalToWorldTransform(tc)
        if not _close(new_w, orig_w):
            bad += 1
            print(f"[flatten]  WORLD-XFORM MISMATCH on {name}")
    print(f"[flatten] transform verification: {len(links) - bad}/{len(links)} links OK")

    st.GetRootLayer().Export(args.dst)
    print(f"[flatten] wrote {args.dst}  (bad transforms: {bad})")
    if bad:
        raise SystemExit("transform verification FAILED")


if __name__ == "__main__":
    main()
