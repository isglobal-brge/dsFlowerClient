import multiprocessing
from pathlib import Path
import queue
import tempfile
import unittest

from public_slots import slot


def hold_slot(root, entered, release):
    with slot(Path(root)) as index:
        entered.put(index)
        release.wait(10)


class PublicSlotTests(unittest.TestCase):
    def test_three_processes_admit_only_two_until_release(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as root:
            entered, release = context.Queue(), context.Event()
            children = [context.Process(target=hold_slot, args=(root, entered, release)) for _ in range(3)]
            try:
                for child in children:
                    child.start()
                self.assertEqual({entered.get(timeout=10), entered.get(timeout=10)}, {0, 1})
                with self.assertRaises(queue.Empty):
                    entered.get(timeout=.5)
                release.set()
                self.assertIn(entered.get(timeout=10), (0, 1))
                for child in children:
                    child.join(10)
                    self.assertEqual(child.exitcode, 0)
            finally:
                release.set()
                for child in children:
                    if child.is_alive():
                        child.terminate()
                    child.join()

    def test_exception_releases_the_slot(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with slot(Path(root)) as index:
                    self.assertEqual(index, 0)
                    raise RuntimeError("fixture")
            with slot(Path(root)) as index:
                self.assertEqual(index, 0)
