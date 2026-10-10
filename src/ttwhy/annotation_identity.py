"""Random human labels and UUIDv4 identities, independent of all image metadata."""

from dataclasses import dataclass, replace
from uuid import uuid4


def random_name() -> str:
    # Lazy import: numerical rendering never imports the naming library.
    try:
        from coolname import generate_slug
    except ImportError as error:
        raise RuntimeError("Selection naming requires coolname; reinstall ttwhy's dependencies") from error
    return generate_slug(3)


@dataclass(frozen=True)
class AnnotationIdentity:
    id: str
    name: str
    name_generated: bool = True

    def renamed(self, name: str) -> "AnnotationIdentity":
        name = name.strip()
        if not name or not name.isprintable():
            raise ValueError("Name must be nonempty, printable text")
        if name == self.name:
            return self
        return replace(self, name=name, name_generated=False)


class IdentityFactory:
    """Avoid generated-name repeats in one viewer; UUID remains the identifier."""

    def __init__(self) -> None:
        self.used_names: set[str] = set()

    def create(self) -> AnnotationIdentity:
        for _ in range(1000):
            name = random_name()
            # coolname's count excludes linking words and some animal names are
            # compound. Keep exactly three short slug components for the footer.
            if (len(name.split("-")) == 3 and len(name) <= 32 and name not in self.used_names
                    and all(part.isascii() and part.isalpha() for part in name.split("-"))):
                self.used_names.add(name)
                return AnnotationIdentity(str(uuid4()), name)
        raise RuntimeError("Could not generate a fresh short name; try selecting again")
