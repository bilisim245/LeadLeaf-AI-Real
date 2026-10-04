"""
Tarla defteri — SQLite (yerel dosya, ek kurulum yok).

Tablolar:
  users        : Telegram kullanicilari (+ il/ilce)
  fields       : kullaniciya ait tarlalar/parseller
  observations : her teshis kaydi (hastalik, guven, baglam, ozet)

Kullanim:
    from bot.db import DB
    db = DB()                       # .env DB_PATH ya da varsayilan
    db.upsert_user(123, "ahmet", il="Antalya", ilce="Serik")
    fid = db.add_field(123, crop="domates", name="alt tarla")
    db.add_observation(123, fid, hastalik="Tomato___Early_blight",
                       guven=0.92, baglam={"gun": 3, "sulama": "yagmurlama"},
                       ozet="Erken yaniklik; bakirli fungisit onerildi.")
    print(db.last_observation(123, fid))
    print(db.recent_cluster("Tomato___Early_blight", "Serik", gun=7))

CLI testi:
    python bot/db.py            # demo veritabani olusturur ve ozet basar
"""
from __future__ import annotations

import os
import json
import sqlite3
import datetime as dt
from typing import Optional

VARSAYILAN_YOL = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "tarla_defteri.sqlite"))

SEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT,
    il          TEXT,
    ilce        TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fields (
    field_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    crop        TEXT,
    name        TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    obs_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(user_id),
    field_id     INTEGER REFERENCES fields(field_id),
    ts           TEXT NOT NULL,
    hastalik     TEXT,
    guven        REAL,
    baglam_json  TEXT,
    ozet         TEXT,
    image_ref    TEXT
);
CREATE INDEX IF NOT EXISTS ix_obs_user ON observations(user_id, field_id, ts);
CREATE INDEX IF NOT EXISTS ix_obs_hastalik ON observations(hastalik, ts);
"""


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class DB:
    def __init__(self, yol: str = VARSAYILAN_YOL):
        self.yol = yol
        os.makedirs(os.path.dirname(os.path.abspath(yol)), exist_ok=True)
        # check_same_thread=False: Streamlit bağlantıyı önbellekte tutuyor ve her rerun farklı
        # bir thread'de çalışabiliyor. Demo tek kullanıcılı, yazmalar sırayla geliyor.
        self.conn = sqlite3.connect(yol, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SEMA)
        self.conn.commit()

    # ---- users ----
    def upsert_user(self, user_id: int, username: str = "",
                    il: Optional[str] = None, ilce: Optional[str] = None) -> None:
        cur = self.conn.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if cur.fetchone():
            alanlar, degerler = [], []
            if username:
                alanlar.append("username=?"); degerler.append(username)
            if il is not None:
                alanlar.append("il=?"); degerler.append(il)
            if ilce is not None:
                alanlar.append("ilce=?"); degerler.append(ilce)
            if alanlar:
                degerler.append(user_id)
                self.conn.execute(f"UPDATE users SET {', '.join(alanlar)} WHERE user_id=?", degerler)
        else:
            self.conn.execute(
                "INSERT INTO users(user_id, username, il, ilce, created_at) VALUES (?,?,?,?,?)",
                (user_id, username, il, ilce, _now()),
            )
        self.conn.commit()

    def get_user(self, user_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    # ---- fields ----
    def add_field(self, user_id: int, crop: str = "", name: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO fields(user_id, crop, name, created_at) VALUES (?,?,?,?)",
            (user_id, crop, name, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_fields(self, user_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM fields WHERE user_id=? ORDER BY field_id", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_or_create_default_field(self, user_id: int, crop: str = "") -> int:
        rows = self.get_fields(user_id)
        if rows:
            return rows[0]["field_id"]
        return self.add_field(user_id, crop=crop, name="varsayilan")

    # ---- observations ----
    def add_observation(self, user_id: int, field_id: Optional[int], hastalik: str,
                        guven: float, baglam: Optional[dict] = None,
                        ozet: str = "", image_ref: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO observations(user_id, field_id, ts, hastalik, guven, baglam_json, ozet, image_ref) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (user_id, field_id, _now(), hastalik, float(guven),
             json.dumps(baglam or {}, ensure_ascii=False), ozet, image_ref),
        )
        self.conn.commit()
        return cur.lastrowid

    def last_observation(self, user_id: int, field_id: Optional[int] = None) -> Optional[dict]:
        if field_id is None:
            row = self.conn.execute(
                "SELECT * FROM observations WHERE user_id=? ORDER BY ts DESC LIMIT 1", (user_id,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM observations WHERE user_id=? AND field_id=? ORDER BY ts DESC LIMIT 1",
                (user_id, field_id),
            ).fetchone()
        return dict(row) if row else None

    def history(self, user_id: int, field_id: Optional[int] = None, limit: int = 10) -> list[dict]:
        if field_id is None:
            rows = self.conn.execute(
                "SELECT * FROM observations WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM observations WHERE user_id=? AND field_id=? ORDER BY ts DESC LIMIT ?",
                (user_id, field_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- bolgesel erken uyari (BONUS) ----
    def recent_cluster(self, hastalik: str, ilce: str, gun: int = 7) -> int:
        """Son `gun` gunde `ilce`'de `hastalik` bildiren FARKLI kullanici sayisi."""
        esik = (dt.datetime.now() - dt.timedelta(days=gun)).isoformat(timespec="seconds")
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT o.user_id) AS n "
            "FROM observations o JOIN users u ON u.user_id = o.user_id "
            "WHERE o.hastalik = ? AND u.ilce = ? AND o.ts >= ?",
            (hastalik, ilce, esik),
        ).fetchone()
        return int(row["n"]) if row else 0

    def users_in_ilce(self, ilce: str) -> list[int]:
        rows = self.conn.execute("SELECT user_id FROM users WHERE ilce=?", (ilce,)).fetchall()
        return [r["user_id"] for r in rows]

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    yol = os.path.join(os.path.dirname(__file__), "_demo_tarla_defteri.sqlite")
    if os.path.exists(yol):
        os.remove(yol)
    db = DB(yol)

    # 3 kullanici, ayni ilce, ayni hastalik -> kumelenme testi
    for uid, uname in [(1, "ahmet"), (2, "mehmet"), (3, "ayse")]:
        db.upsert_user(uid, uname, il="Antalya", ilce="Serik")
        fid = db.get_or_create_default_field(uid, crop="domates")
        db.add_observation(uid, fid, hastalik="Tomato___Early_blight", guven=0.9,
                           baglam={"gun": 3, "sulama": "yagmurlama"},
                           ozet="Erken yaniklik; bakirli fungisit onerildi.")

    print("kullanici 1 son kayit :", db.last_observation(1))
    print("Serik / erken yaniklik / 7 gun -> farkli kullanici:",
          db.recent_cluster("Tomato___Early_blight", "Serik", gun=7))
    print("Serik'teki kullanicilar :", db.users_in_ilce("Serik"))
    db.close()
    os.remove(yol)
    print("OK — db.py calisiyor.")
