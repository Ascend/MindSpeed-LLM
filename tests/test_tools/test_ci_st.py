import warnings
import pytest
from mindspeed_llm import megatron_adaptor
from tests.test_tools.acquire_json import transfer_logs_as_json, read_json

MEMO_INFO = "memo info"
THROUGHPUT = "throughput"
LOSS = "lm loss"
TIME_INFO = "time info"
GRAD_NORM_INFO = "grad norm"

WARM_UP = 5


class TestMargin:
    _MARGIN_NAME = " margin"
    loss = 0.02
    grad_norm = 0.1
    memory = 0.1
    throughput = 0.05
    time = 0.05

    @classmethod
    def refresh_margin_from_json(cls, json_obj):
        cls.loss = json_obj.get(LOSS + cls._MARGIN_NAME, cls.loss)
        cls.memory = json_obj.get(MEMO_INFO + cls._MARGIN_NAME, cls.memory)
        cls.throughput = json_obj.get(THROUGHPUT + cls._MARGIN_NAME, cls.throughput)
        cls.time = json_obj.get(TIME_INFO + cls._MARGIN_NAME, cls.time)
        cls.grad_norm = json_obj.get(GRAD_NORM_INFO + cls._MARGIN_NAME, cls.grad_norm)


class TestCIST:

    margin_loss = 0.02
    margin_grad_norm = 0.1
    margin_throughput_percent = 0.05
    margin_memory_percent = 0.1
    margin_time_percent = 0.05

    def _get_baseline(self, baseline_json):
        # acquire expected results
        self.expected = read_json(baseline_json)
        TestMargin.refresh_margin_from_json(self.expected)

    def _get_actual(self, generate_log, generate_json):
        # acquire actual results
        transfer_logs_as_json(generate_log, generate_json)
        self.actual = read_json(generate_json)
    
    def _test_helper(self, test_obj):
        """
        Core test function

        Args:
            test_obj: the object we want to test compare.
            test_type: deterministic or approximate, default is None.

        Here we temperally test `lm loss`, 'throughput' , `allocated memory` and `elapsed time per iteration`
        """
        comparison_base = {
            LOSS: self._compare_lm_loss,
            MEMO_INFO: self._compare_memory,
        }
        
        comparison_throughput = {
            THROUGHPUT: self._compare_throughput,
        }
        
        comparison_time = {
            TIME_INFO: self._compare_time,
        }

        comparison_grad_norm = {
            GRAD_NORM_INFO: self._compare_grad_norm,
        }

        comparison_selection = {**comparison_base}
        if "time info" in self.expected:
            comparison_selection = {**comparison_selection, **comparison_time}

        if "grad norm" in self.expected:
            comparison_selection = {**comparison_selection, **comparison_grad_norm}

        if "throughput" in self.actual:
            comparison_selection = {**comparison_selection, **comparison_throughput}
        
        if test_obj in comparison_selection:
            expected_list = self.expected[test_obj]
            if not expected_list:
                return
            print(f"===================== Begin comparing {test_obj} ===================")
            actual_list = self.actual[test_obj]
            print(f"The list of expected values: {expected_list}")
            print(f"The list of actual values: {actual_list}")
            # Check if lists exist and are non-empty
            if not actual_list:
                raise ValueError(f"Actual list for {test_obj} is empty or not found. Maybe program has failed! Check it.")

            # Check if lists have the same length
            if len(expected_list) != len(actual_list):
                raise ValueError(f"Actual lengths of the lists for {test_obj} do not match. Maybe program has failed! Check it.")

            compare_func = comparison_selection[test_obj]
            compare_func(expected_list, actual_list)
        else:
            warnings.warn(f"The metric of {test_obj} is not selected and will be skipped.", RuntimeWarning, stacklevel=2)
            
    def _compare_lm_loss(self, expected_list, actual_list):
        # Because "deterministic computation" affects the throughput, so we just test
        # lm loss in case of approximation.
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for lm loss")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.loss),\
            f"The loss at step {step} should be approximate to {expected_val} but it is {actual_val}."

    
    def _compare_grad_norm(self, expected_list, actual_list):
        # Because "deterministic computation" affects the throughput, so we just test
        # grad norm in case of approximation.
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for grad norm")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.grad_norm),\
            f"The grad norm at step {step} should be approximate to {expected_val} but it is {actual_val}."

            
    def _compare_throughput(self, expected_list, actual_list):
        # First few iterations might take a little longer. So we take the last 70 percent of the timings
        try:
            expected_avg_throughput = sum(expected_list[WARM_UP:]) / (len(expected_list) - WARM_UP)
            actual_avg_throughput = sum(actual_list[WARM_UP:]) / (len(actual_list) - WARM_UP)
        except:
            raise ZeroDivisionError
        
        assert actual_avg_throughput >= expected_avg_throughput or \
            abs(actual_avg_throughput - expected_avg_throughput) / expected_avg_throughput <= TestMargin.throughput, \
            f"The actual avg throughput {actual_avg_throughput} degradate expected avg throughput {expected_avg_throughput}"


    def _compare_time(self, expected_list, actual_list):
        try:
            expected_avg_time = sum(expected_list[WARM_UP:]) / (len(expected_list) - WARM_UP)
            actual_avg_time = sum(actual_list[WARM_UP:]) / (len(actual_list) - WARM_UP)
        except:
            raise ZeroDivisionError
        
        assert actual_avg_time <= expected_avg_time or \
               abs(actual_avg_time - expected_avg_time) / expected_avg_time <= TestMargin.time, \
            f"The actual avg time ({actual_avg_time} ms) was slower than the expected ({expected_avg_time} ms)."


    def _compare_memory(self, expected_list, actual_list):
        for i, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            assert actual_val["allocated memory"] <= expected_val["allocated memory"] or \
                abs(actual_val["allocated memory"] - expected_val["allocated memory"]) / expected_val["allocated memory"] <= TestMargin.memory, \
                f'The actual memory {actual_val["allocated memory"]} seems to be abnormal compare to expected {expected_val["allocated memory"]}.'
            
            assert actual_val["max allocated memory"] <= expected_val["max allocated memory"] or \
                abs(actual_val["max allocated memory"] - expected_val["max allocated memory"]) / expected_val["max allocated memory"] <= TestMargin.memory, \
                f'The actual max memory {actual_val["max allocated memory"]} seems to be abnormal compare to expected {expected_val["max allocated memory"]}.'


    def test_lm_loss_approx(self, baseline_json, generate_log, generate_json):
        # expected training loss curve at different global steps.
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("lm loss")


    def test_grad_norm_approx(self, baseline_json, generate_log, generate_json):
        # expected training loss curve at different global steps.
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("grad norm")


    def test_througpout(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("throughput")


    def test_time(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("time info")


    def test_allocated_memory(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("memo info")
