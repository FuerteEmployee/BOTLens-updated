"""
One-off: export the faces registered on the OLD kiosk (its own SQLite file and
photo folders) so they can be imported into the B.O.T backend, where faces live
now. Read-only: nothing in the old file or folders is changed.

    python export_faces.py <old attendance.db> <photo dir> <out.json>

For each active employee linked to a backend employee, every photo in their
folder is re-encoded with the same SFace model the kiosk matches with, and one
small thumbnail is cut from the first photo with a face. The output is then
checked and imported by backend/scripts/import_lens_faces.js (dry run first).
"""
import base64
import json
import os
import sqlite3
import sys

import engine as face


def main(db_file, photo_dir, out_file):
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    rows = con.execute(
        "select id, name, photo_path, face_encoding, external_admin_id, external_user_id "
        "from employees where is_active = 1"
    ).fetchall()
    out, skipped = [], []
    for emp_id, name, photo_path, stored, admin_id, user_id in rows:
        if not admin_id or not user_id:
            skipped.append({"name": name, "reason": "not linked to a B.O.T employee"})
            continue
        folder = None
        if photo_path:
            # Stored with whatever separators the kiosk ran under; re-root on photo_dir.
            leaf = os.path.basename(os.path.dirname(photo_path.replace("\\", "/")))
            cand = os.path.join(photo_dir, leaf)
            folder = cand if os.path.isdir(cand) else None
        vectors, thumb_src = [], None
        if folder:
            for f in sorted(os.listdir(folder)):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                with open(os.path.join(folder, f), "rb") as fh:
                    vec, src = face.encode_photo(fh.read())
                if vec is not None:
                    vectors.append([float(x) for x in vec])
                    thumb_src = thumb_src or src
                if len(vectors) >= 12:
                    break
        if not vectors and stored:
            try:  # the single vector the old kiosk kept in its database
                data = json.loads(stored)
                if data and isinstance(data[0], (int, float)) and len(data) == 128:
                    vectors.append([float(x) for x in data])
            except ValueError:
                pass
        if not vectors:
            skipped.append({"name": name, "reason": "no usable face photo"})
            continue
        jpg = face.thumbnail_jpeg(thumb_src)
        out.append({
            "kioskName": name,
            "adminId": admin_id,
            "employeeId": user_id,
            "embeddings": vectors,
            "thumbnail": ("data:image/jpeg;base64," + base64.b64encode(jpg).decode()) if jpg else None,
            "photos": len(vectors),
        })
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump({"faces": out, "skipped": skipped, "model": face.MODEL_VERSION}, fh)
    print(f"exported {len(out)} faces, skipped {len(skipped)}")
    for s in skipped:
        print(f"  skipped {s['name']}: {s['reason']}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
