import logging
from pathlib import Path
from typing import List, Tuple

from opentuner import resultsdb
from opentuner.resultsdb.models import Configuration, Result, TuningRun
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.expression import func

from ._logging import PathLike

msg_logger = logging.getLogger(__name__)


def read_opentuner_db(filepath_or_uri: PathLike) -> List[Tuple[Result, Configuration]]:
    if "://" in filepath_or_uri:
        uri = filepath_or_uri
    else:
        filepath = Path(filepath_or_uri)
        uri = f"sqlite:///{filepath}"
    try:
        _, sess = resultsdb.connect(uri)
    except Exception as e:
        msg_logger.error("Failed to load database: %s", filepath_or_uri, exc_info=True)
        raise e
    session: Session = sess()
    latest_run_id = session.query(func.max(TuningRun.id)).all()[0][0]
    run_results = (
        session.query(Result, Configuration)
        .filter_by(tuning_run_id=latest_run_id)
        .filter(Result.configuration_id == Configuration.id)
        .all()
    )
    return run_results
