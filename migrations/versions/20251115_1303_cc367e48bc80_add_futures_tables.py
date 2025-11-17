"""add futures tables

Revision ID: cc367e48bc80
Revises: 62279d85acef
Create Date: 2025-11-15 13:03:05.165326+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cc367e48bc80'
down_revision: Union[str, None] = '62279d85acef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
