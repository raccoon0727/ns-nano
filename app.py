# app.py
from flask import Flask, jsonify, abort, request, send_file
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL (or DATABASE_URL) environment variable.")
    # Normalize old 'postgres://' scheme to 'postgresql://'
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
    )
    return _engine

def create_app():
    app = Flask(__name__)

    @app.get("/", endpoint="health")
    def health():
        return "<p>Server working!</p>"

    @app.get("/img", endpoint="show_img")
    def show_img():
        return send_file("amygdala.gif", mimetype="image/gif")

    @app.get("/terms/<term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        return term

    @app.get("/locations/<coords>/studies", endpoint="locations_studies")
    def get_studies_by_coordinates(coords):
        x, y, z = map(int, coords.split("_"))
        return jsonify([x, y, z])

    # ------------------------------------------------------------------
    # Functional dissociation endpoints
    # ------------------------------------------------------------------
    def _limit():
        try:
            return max(1, min(int(request.args.get("limit", 1000)), 10000))
        except ValueError:
            abort(400, "limit must be an integer")

    def _metadata_key(conn):
        cols = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'ns' AND table_name = 'metadata'"))}
        return "study_id" if "study_id" in cols else "id"

    def _studies_response(ids_sql, params, query_info):
        eng = get_engine()
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))
                key = _metadata_key(conn)
                rows = conn.execute(text(f"""
                    WITH hits AS ({ids_sql})
                    SELECT h.study_id,
                           (to_jsonb(m) - 'fts') AS metadata
                    FROM hits h
                    LEFT JOIN ns.metadata m ON m.{key}::text = h.study_id
                    ORDER BY h.study_id
                    LIMIT :lim
                """), {**params, "lim": _limit()}).mappings().all()
                total = conn.execute(text(
                    f"SELECT COUNT(*) FROM ({ids_sql}) t"), params).scalar()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({
            "ok": True,
            **query_info,
            "count": total,
            "returned": len(rows),
            "studies": [{"study_id": r["study_id"], "metadata": r["metadata"]} for r in rows],
        })

    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        """Studies that mention term_a but NOT term_b (A \ B)."""
        # 去除頭尾空白、轉小寫，並將空格轉為底線
        clean_a = term_a.strip().lower().replace(" ", "_")
        clean_b = term_b.strip().lower().replace(" ", "_")
        if not clean_a or not clean_b:
            abort(400, "both terms are required")

        # 匹配包含前綴（如 terms_abstract_tfidf__posterior_cingulate）的格式
        pat_a = f"%{clean_a}%"
        pat_b = f"%{clean_b}%"

        ids_sql = """
            SELECT DISTINCT study_id FROM ns.annotations_terms
            WHERE LOWER(term) LIKE :pat_a
              AND study_id NOT IN (
                  SELECT study_id FROM ns.annotations_terms WHERE LOWER(term) LIKE :pat_b
              )
        """
        return _studies_response(
            ids_sql,
            {"pat_a": pat_a, "pat_b": pat_b},
            {"type": "terms", "in": term_a, "not_in": term_b}
        )

    @app.get("/dissociate/locations/<loc_a>/<loc_b>", endpoint="dissociate_locations")
    def dissociate_locations(loc_a, loc_b):
        """Studies reporting a peak near [x1,y1,z1] but NOT near [x2,y2,z2].

        'Near' = within `r` mm (default 10, override with ?r=) in 3D.
        """
        def parse(s):
            try:
                x, y, z = (float(v) for v in s.split("_"))
            except ValueError:
                abort(400, f"bad coordinate '{s}', expected x_y_z")
            return x, y, z
        try:
            r = float(request.args.get("r", 10))
        except ValueError:
            abort(400, "r must be a number")
        ax, ay, az = parse(loc_a)
        bx, by, bz = parse(loc_b)
        ids_sql = """
            SELECT DISTINCT study_id FROM ns.coordinates
            WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:ax,:ay,:az), 4326), :r)
              AND study_id NOT IN (
                  SELECT study_id FROM ns.coordinates
                  WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:bx,:by,:bz), 4326), :r))
        """
        return _studies_response(
            ids_sql,
            {"ax": ax, "ay": ay, "az": az, "bx": bx, "by": by, "bz": bz, "r": r},
            {"type": "locations", "in": [ax, ay, az], "not_in": [bx, by, bz], "radius_mm": r})

    @app.get("/test_db", endpoint="test_db")
    
    def test_db():
        eng = get_engine()
        payload = {"ok": False, "dialect": eng.dialect.name}

        try:
            with eng.begin() as conn:
                # Ensure we are in the correct schema
                conn.execute(text("SET search_path TO ns, public;"))
                payload["version"] = conn.exec_driver_sql("SELECT version()").scalar()

                # Counts
                payload["coordinates_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.coordinates")).scalar()
                payload["metadata_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.metadata")).scalar()
                payload["annotations_terms_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.annotations_terms")).scalar()

                # Samples
                try:
                    rows = conn.execute(text(
                        "SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z FROM ns.coordinates LIMIT 3"
                    )).mappings().all()
                    payload["coordinates_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["coordinates_sample"] = []

                try:
                    # Select a few columns if they exist; otherwise select a generic subset
                    rows = conn.execute(text("SELECT * FROM ns.metadata LIMIT 3")).mappings().all()
                    payload["metadata_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["metadata_sample"] = []

                try:
                    rows = conn.execute(text(
                        "SELECT study_id, contrast_id, term, weight FROM ns.annotations_terms LIMIT 3"
                    )).mappings().all()
                    payload["annotations_terms_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["annotations_terms_sample"] = []

            payload["ok"] = True
            return jsonify(payload), 200

        except Exception as e:
            payload["error"] = str(e)
            return jsonify(payload), 500

    return app

# WSGI entry point (no __main__)
app = create_app()
