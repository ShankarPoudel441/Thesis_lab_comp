from influxdb import InfluxDBClient
import pandas as pd
import dataclasses
import numpy as np
from scipy import signal

###

@dataclasses.dataclass
class vibration:
    """vibration datapoint.

    Attributes:
        timestamp (int): timestamp of vibration measurement.
        data (list): recorded vibration data points.
    """

    timestamp: int
    data: list

    @classmethod
    def from_dict(self, d):
        """Create data object from dictionary object."""
        return self(**d)

    def to_dict(self):
        """Return dictionary object."""
        return dataclasses.asdict(self)

    def __add__(self, x):
        """Define custom addition rule."""
        self.data.extend(x.data)
        return self
    
def convert_to_vibration(df):
    """Compute vibration objects from dataframe"""
    vibration_list = []
    for i in range(127, 128 * (len(df) // 128), 128):
        vibration_list.append(
            vibration(
                timestamp=df.loc[i, "time"],
                data=df.loc[i - 127 : i, "vibration"].values,
            )
        )
    return vibration_list

###

@dataclasses.dataclass
class filterOp:
    """filtered op dsatapoints.

    Attributes:
        timestamp (int): time of first sample
        y: list of seven float values : [y0,y1,y2,y3,y4,y5,y6]
    """

    timestamp: int
    y0: list
    y1: list
    y2: list
    y3: list
    y4: list
    y5: list
    y6: list

    @classmethod
    def from_dict(self, d):
        """Create data object from dictionary object."""
        return self(**d)

    def to_dict(self):
        """Return the values in dictionary form"""
        return dataclasses.asdict(self)
    
    
@dataclasses.dataclass
class filterBank:
    """Filter bank data class to store coefficients

    Initialized with a fundamental frequency

    Args:
        fundFreq (float): Fundamental frequency for filter bank.
    """

    def __init__(self,fundFreq: float = 62.5, fs: int =1000):
        self.F_ORDER = 4
        self._fundFreq = fundFreq
        self._fs = fs
        self._zf = np.zeros([7, 8])
        self._createFilterCoeff()

    @classmethod
    def from_dict(self, d):
        """Create datapoint from dictionary object."""
        return self(**d)

    def to_dict(self):
        """Create dictionary object."""
        return dataclasses.asdict(self)

    @property
    def fundFreq(self):
        return self._fundFreq

    @fundFreq.setter
    def fundFreq(self, fundFreq):
        try:
            if isinstance(fundFreq, float):
                pass
            else:
                fundFreq = float(fundFreq)
            assert isinstance(fundFreq, float)
        except TypeError:
            raise FilterClassError("Fundamental Frequency value must be a float")
        else:
            self._fundFreq = fundFreq
            self._createFilterCoeff()

    def _createFilterCoeff(self):
        """Generate filter coefficients based on fundamental frequency.

        Creates a 7-tap filter bank using bandpass filters located at the fundamental frequency
        and 6 harmonics.

        Filter coefficients are saved as class attributes for later processing.
        """
        self.b0, self.a0 = signal.butter(
            N=self.F_ORDER,
            Wn=[self.fundFreq - 1, self.fundFreq + 1],
            btype="bandpass",
            fs=self._fs,
        )
        self.b1, self.a1 = signal.butter(
            N=self.F_ORDER,
            Wn=[2 * self.fundFreq - 3, 2 * self.fundFreq + 3],
            btype="bandpass",
            fs=self._fs,
        )
        self.b2, self.a2 = signal.butter(
            self.F_ORDER,
            [3 * self.fundFreq - 5, 3 * self.fundFreq + 5],
            "bandpass",
            fs=self._fs,
        )
        self.b3, self.a3 = signal.butter(
            self.F_ORDER,
            [4 * self.fundFreq - 5, 4 * self.fundFreq + 5],
            "bandpass",
            fs=self._fs,
        )
        self.b4, self.a4 = signal.butter(
            self.F_ORDER,
            [5 * self.fundFreq - 5, 5 * self.fundFreq + 5],
            "bandpass",
            fs=self._fs,
        )
        self.b5, self.a5 = signal.butter(
            self.F_ORDER,
            [6 * self.fundFreq - 5, 6 * self.fundFreq + 5],
            "bandpass",
            fs=self._fs,
        )
        self.b6, self.a6 = signal.butter(
            self.F_ORDER,
            [7 * self.fundFreq - 5, 7 * self.fundFreq + 5],
            "bandpass",
            fs=self._fs,
        )

    def filter(self, rawVibration):
        """Filter a vibration signal and return the result as a dictionary"""
        (y0, z0) = signal.lfilter(self.b0, self.a0, x=rawVibration.data, zi=self._zf[0])
        (y1, z1) = signal.lfilter(self.b1, self.a1, x=rawVibration.data, zi=self._zf[1])
        (y2, z2) = signal.lfilter(self.b2, self.a2, x=rawVibration.data, zi=self._zf[2])
        (y3, z3) = signal.lfilter(self.b3, self.a3, x=rawVibration.data, zi=self._zf[3])
        (y4, z4) = signal.lfilter(self.b4, self.a4, x=rawVibration.data, zi=self._zf[4])
        (y5, z5) = signal.lfilter(self.b5, self.a5, x=rawVibration.data, zi=self._zf[5])
        (y6, z6) = signal.lfilter(self.b6, self.a6, x=rawVibration.data, zi=self._zf[6])
        self._zf = np.array([z0, z1, z2, z3, z4, z5, z6])
        return filterOp.from_dict(
            {
                "timestamp": rawVibration.timestamp,
                "y0": y0,
                "y1": y1,
                "y2": y2,
                "y3": y3,
                "y4": y4,
                "y5": y5,
                "y6": y6,
            }
        )

###

@dataclasses.dataclass
class Power:
    """Power obejct to hold results of power calculations.

    Args:
        timestamp: int
        total_power: float
        p0: float
        p1: float
        p2: float
        p3: float
        p4: float
        p5: float
        p6: float
    """

    timestamp: int
    total_power: float
    p0: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    p6: float

    @classmethod
    def from_dict(self, d):
        """Create data object from dictionary object."""
        return self(**d)

    def to_dict(self):
        """Return dictionary object."""
        return dataclasses.asdict(self)

@dataclasses.dataclass
class PowerModule:
    """Calculate power from vibrations.

    Args:
        N (int): Window Size (default=256)
        stepSize (int): Sliding window step-size (default=128)
        fs (int): Sample Frequency of vibration data [Hz] (default=1000)
    """

    fundFreq: float = 45.0
    N: int = 128
    stepSize: int = 128
    fs: float = 1000
    mean: np.array = np.array([0, 0, 0, 0, 0, 0, 0])
    std: np.array = np.array([1, 1, 1, 1, 1, 1, 1])
    K: float = 1.0 / 0.5364

    @classmethod
    def from_dict(self, d):
        """Create datapoint from a dictionary object."""
        return self(**d)

    def to_dict(self):
        """Turn datapoint into a dictionary object."""
        return dataclasses.asdict(self)

    def compute(self, filter_output, normalize=True):
        powerBands = self._computePowerbands(filter_output)
        total_power = self._computeTotalPower(powerBands)
        powerBands.update(total_power)
        # print(self.mean,self.std)
        # print(normalize)
        if normalize:
            # print("Normalizing")
            powerBands = self._normalizePower(powerBands)
        else:
            pass
            # print("not norma")
        powerBands.update({"timestamp": filter_output.timestamp})
        return Power.from_dict(powerBands)

    def _normalizePower(self, powerBands):
        """Normalize powerbands according to calibration file"""
        dict_names = ["total_power"]
        [dict_names.append(f"p{i}") for i in range(len(self.mean))]

        for i in range(len(self.mean)):
            powerBands[dict_names[i]] -= self.mean[i]
            powerBands[dict_names[i]] /= self.std[i]
            powerBands[dict_names[i]] = np.clip(powerBands[dict_names[i]], -2, 2)
        return powerBands

    def _computeTotalPower(self, x):
        """Compute total power from powerbands."""
        f = (
            np.array(
                [
                    self.fundFreq,
                    2 * self.fundFreq,
                    3 * self.fundFreq,
                    4 * self.fundFreq,
                    5 * self.fundFreq,
                    6 * self.fundFreq,
                    7 * self.fundFreq,
                ]
            )
            ** 2
        )
        return {
            "total_power": np.matmul(
                f, [x["p0"], x["p1"], x["p2"], x["p3"], x["p4"], x["p5"], x["p6"]]
            )
            / 1000000  # Divide by 1 million is from Paloma's experiments
        }

    def _computePowerbands(self, filterOutput):
        """Calculate powerbands from 7-tap filter output.

        Args:
            filterOutput (filterOp): a filter output object
        Returns:
            p (dict): a dictionary of power outputs
        """
        p0 = self._computePower(filterOutput.y0)
        p1 = self._computePower(filterOutput.y1)
        p2 = self._computePower(filterOutput.y2)
        p3 = self._computePower(filterOutput.y3)
        p4 = self._computePower(filterOutput.y4)
        p5 = self._computePower(filterOutput.y5)
        p6 = self._computePower(filterOutput.y6)
        powerBand = {
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5,
            "p6": p6,
        }
        return powerBand

    def _computePower(self, x):
        """Calculate the power of a single input signal.

        For a signal the size of 256, two power points will be computed.
        We will only report the first, which is at t=128ms.

        Args:
            x (np.array): A filtered vibration signal.
        Returns:
            power (np.array): Signal power
        """
        window = np.hamming(self.N)

        y0 = self.K * np.multiply(window, x)
        power = (np.linalg.norm(y0) ** 2) / self.N
        return power

def estimate_fundFreq(vib_data_pd):

    frequencies, times, spectrogram = signal.spectrogram(
        vib_data_pd.vibration.values,
        1000,
        "boxcar",
        nperseg=256,
        noverlap=128,
        scaling="density",
        mode="magnitude",
    )

    S_mean = np.mean(spectrogram, axis=1)
    fundFreq = frequencies[np.argmax(S_mean)]
    try:
        print(f"fundFreq:{fundFreq}")
    except:  # Encountered during unit-test
        print ("Fund freq difficulty")
    return fundFreq


def get_mean_std(power_O):
    """
    get the list of power object and get the mean and std with each of those
    
    O/P:mean= array([8.29435293e-08, 2.80233810e-05, 3.27951217e-05, 2.64883938e-05,
        1.48420347e-05, 9.54587842e-06, 6.67679403e-06, 4.99720963e-06]),
        std = array([3.78414996e-06, 8.95038345e-04, 1.50523947e-03, 1.26197150e-03,
        7.01275731e-04, 4.46176335e-04, 3.07947925e-04, 2.26973899e-04])}
    """
    data=pd.DataFrame([s.to_dict() for s in power_O])
    print("Length of full objects",len(power_O), "\nLen of created ones", len(data["p0"]))
    mean=data.mean()
    std=data.std()
    return(mean,std)

