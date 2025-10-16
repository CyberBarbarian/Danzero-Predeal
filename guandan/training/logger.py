import os, csv, time, json, logging
from typing import Dict, Any, Optional
import numbers
try:
    from torch.utils.tensorboard import SummaryWriter  # Preferred
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

class Logger:
    def __init__(self, log_dir: str, xpid: str, enable_tensorboard: bool = True):
        self.base = os.path.join(log_dir, xpid)
        os.makedirs(self.base, exist_ok=True)
        self.fields_path = os.path.join(self.base, 'fields.csv')
        self.logs_path = os.path.join(self.base, 'logs.csv')
        self.meta_path = os.path.join(self.base, 'meta.json')
        self.tb_dir = os.path.join(self.base, 'tb')
        self.tick = 0
        self.fieldnames = ['_tick','_time']
        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger('training.logger')
        self.writer: Optional[SummaryWriter] = None
        if enable_tensorboard and SummaryWriter is not None:
            os.makedirs(self.tb_dir, exist_ok=True)
            try:
                # Use shorter flush interval so curves show up quickly during long runs
                self.writer = SummaryWriter(log_dir=self.tb_dir, flush_secs=10, max_queue=20)
            except Exception as e:  # pragma: no cover
                self._log.warning(f"Failed to initialize SummaryWriter: {e}")

    def log(self, data: Dict[str, Any]):
        data = dict(data)
        data['_tick'] = self.tick
        data['_time'] = time.time()
        self.tick += 1
        # update fields
        new = False
        for k in data:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
                new = True
        if new or self.tick == 1:
            with open(self.fields_path,'w',newline='') as f:
                csv.writer(f).writerow(self.fieldnames)
        if self.tick == 1:
            if not os.path.exists(self.logs_path):
                with open(self.logs_path,'a') as f:
                    f.write('# ' + ','.join(self.fieldnames) + '\n')
        with open(self.logs_path,'a',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)
        # TensorBoard scalar logging for numeric values
        if self.writer is not None:
            for key, value in data.items():
                if key.startswith('_'):
                    continue
                # Accept Python numbers and NumPy scalar types
                if isinstance(value, numbers.Number):
                    try:
                        self.writer.add_scalar(key, value, global_step=self.tick)
                    except Exception:
                        pass
            # Light flush to ensure timely visibility in TensorBoard
            try:
                self.writer.flush()
            except Exception:
                pass

    def save_meta(self, meta: Dict[str, Any]):
        with open(self.meta_path,'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # Also log meta as text to TensorBoard for reference
        if self.writer is not None:
            try:
                self.writer.add_text('meta', json.dumps(meta, ensure_ascii=False, indent=2), global_step=0)
            except Exception:
                pass

    def close(self):
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass

__all__ = ['Logger']

