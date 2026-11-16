"""The namespace this plugin's entries live in, registered once, at import.

The naming rule, in one place (nothing here renames anything: it is how MIME names things today):

- an **entry** is `<namespace>.<name>`, with a dot: `softtissue.notched_strip`;
- a **record** (an asset) is `<namespace>_<name>`, with an underscore: `softtissue_fibrin_plasma`;
- an **instance** of an entry on a thing is `<entry>:<thing>`, with a colon: `softtissue.grip_force:strip`.

The namespace appears in six places, filled from one value by `mime scaffold plugin`: the `mime.entries` and `mime.assets`
entry points in pyproject.toml, the line below, the entry's name, the record's name and the instance in examples/.
"""

from mime.vocabulary import Interface, ReadKind, define, register_namespace

NAMESPACE, VERSION = "softtissue", "0.1.0"
register_namespace(NAMESPACE, VERSION)

GRIP_FORCE = define(Interface("softtissue.specimen.grip_force", ReadKind.STATE, "N", None, (), magnitude=0.1,
                              description="the force the specimen's grip carries along the pull"))
GRIP_DISPLACEMENT = define(Interface("softtissue.specimen.grip_displacement", ReadKind.STATE, "m", None, (),
                                     magnitude=1e-3, description="the grip's displacement from the start, total"))
CRACK_LENGTH = define(Interface("softtissue.specimen.crack_length", ReadKind.STATE, "m", None, (), magnitude=1e-3,
                                description="the phase field's crack length: its surface energy over Gc and thickness"))
