#!/usr/bin/env python3
import sqlite3
from dataclasses import dataclass, field
from typing import Any


# Define model
@dataclass
class Contact:
    id: int
    name: str
    email: str = field(default_factory=str)

    # Implement validation
    def validate(self):
        if not self.name or not self.email:
            raise ValueError("Name and email cannot be empty")
        return True


# Connect to SQLite database
with sqlite3.connect("contacts.db") as conn:
    cur = conn.cursor()

    cur.execute(
        """CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )"""
    )
    conn.commit()

    # Insert data
    contact = Contact(id=1, name="Jane Doe", email="jane.doe@fakemail.com")
    if contact.validate():
        cur.execute(
            "INSERT INTO contacts VALUES (?, ?, ?)",
            (contact.id, contact.name, contact.email),
        )
        conn.commit()

    # Querying the database
    for row in cur.execute("SELECT * FROM contacts"):
        print(row)

    # Handling validation errors
    try:
        bad_contact = Contact(id=2, name="", email="john.doe@fakemail.com")
        if bad_contact.validate():
            cur.execute(
                "INSERT INTO contacts VALUES (?, ?, ?)",
                (bad_contact.id, bad_contact.name, bad_contact.email),
            )
            conn.commit()
    except ValueError as err:
        print(err)
