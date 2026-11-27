"""Test-only helpers: lookups the real code never needs."""
from lifecompass import store


def get_image(image_id):
    """One image row by id, or None."""
    with store._connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    return store._row_to_image(row) if row else None
