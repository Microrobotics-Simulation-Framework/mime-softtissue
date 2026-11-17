"""The namespace this plugin's entries live in, registered once, at import, and the interfaces its entries publish and
accept (as ``mime.vocabulary`` holds MIME's), so that a dynamics, the law that loads it and the observation that reads it
name one definition.

The naming rule, in one place (nothing here renames anything: it is how MIME names things today):

- an **entry** is `<namespace>.<name>`, with a dot: `softtissue.notched_strip`;
- a **record** (an asset) is `<namespace>_<name>` or `<namespace>.<name>`: the plugin's first records used the
  underscore, its later ones the dot; both are read today, and dotted record names wait for MIME's one-`asset()` change;
- an **instance** of an entry on a thing is `<entry>:<thing>`, with a colon: `softtissue.grip_force:strip`.

The namespace appears in six places, filled from one value by `mime scaffold plugin`: the `mime.entries` and `mime.assets`
entry points in pyproject.toml, the line below, the entry's name, the record's name and the instance in examples/.
"""

from mime.vocabulary import Interface, ReadKind, TargetKind, define, register_namespace

NAMESPACE, VERSION = "softtissue", "0.1.0"
register_namespace(NAMESPACE, VERSION)

# --- the notched strip (``softtissue.notched_strip``, read by ``softtissue.grip_force``) ---------------------------------

GRIP_FORCE = define(Interface("softtissue.specimen.grip_force", ReadKind.STATE, "N", None, (), magnitude=0.1,
                              description="the force the specimen's grip carries along the pull"))
GRIP_DISPLACEMENT = define(Interface("softtissue.specimen.grip_displacement", ReadKind.STATE, "m", None, (),
                                     magnitude=1e-3, description="the grip's displacement from the start, total"))
CRACK_LENGTH = define(Interface("softtissue.specimen.crack_length", ReadKind.STATE, "m", None, (), magnitude=1e-3,
                                description="the phase field's crack length: its surface energy over Gc and thickness"))

# --- the gel block (``softtissue.gel_block``, loaded by ``softtissue.tip_contact``, read by the tip's observations) ------

#: The tip's numbers the law hands the block: pose (7), the part's centre (3) and rotation (9, rows) in the body's
#: frame, its region's parameters (``N_REGION``, in the shape's ``parameter_names`` order), friction, adhesive shear
#: (Pa) and the penalty's stiffness (Pa/m). An interface's shape is fixed at definition, so ``N_REGION`` is the most
#: parameters a registered tip shape may have (LF-1's seam 1c, met again).
N_REGION = 4
N_INDENTER = 7 + 3 + 9 + N_REGION + 3
INDENTER = define(Interface("softtissue.block.indenter", TargetKind.EQUALITY_CONSTRAINT, "1", "world", (N_INDENTER,),
                            magnitude=1e-3, description="the tip pressing the block: its pose, part frame, shape "
                                                        "parameters and interface numbers, as one vector"))
TIP_FORCE = define(Interface("softtissue.block.tip_force", ReadKind.STATE, "N", None, (), magnitude=1e-2,
                             description="the force the block puts on the tip along +z (positive: pushing it out)"))
TIP_DEPTH = define(Interface("softtissue.block.tip_depth", ReadKind.STATE, "m", None, (), magnitude=1e-3,
                             description="the tip's lowest point below the block's undeformed top face"))
CRACK_AREA = define(Interface("softtissue.block.crack_area", ReadKind.STATE, "m^2", None, (), magnitude=1e-6,
                              description="the crack's area: the phase field's surface energy over the toughness"))
