"""List of univariate non-normalized datasets (UCR) which are to be normalized upon loading"""
univar_non_normalized_datasets = [
    'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',  'DodgerLoopGame', 'DodgerLoopWeekend', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'PickupGestureWiimoteZ', 'PLAID',
    'ShakeGestureWiimoteZ', 'Fungi', 'Rock', 'EOGVerticalSignal', 'EOGHorizontalSignal', 'Chinatown', 'SemgHandSubjectCh2', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'GunPointMaleVersusFemale', 'PigArtPressure', 'PigCVP', 'GunPointOldVersusYoung', 'GunPointAgeSpan', 'HouseTwenty'
]

"""
List of univariate datasets (UCR) which are to be loaded from separate folder, where missing values have been interpolated. N.B the README of 'Chinatown' says it has missing values, but the data files contain no NaN values. The 'Chinatown' dataset isn't present in the folder 'Missing_value_and_variable_length_datasets_adjusted' either, so the dataset isn't included in this list. It is treated as a regular dataset.
"""
univar_interpolated_values_datasets = [
    'MelbournePedestrian', 'DodgerLoopDay', 'DodgerLoopWeekend', 'DodgerLoopGame'
]

"""List of multivariate datasets from UEA archive"""
multivariate_datasets = [
    'ArticularyWordRecognition', 'ERing', 'HandMovementDirection', 'Libras', 'RacketSports',
    'AtrialFibrillation', 'EigenWorms', 'Handwriting', 'MotorImagery', 'SelfRegulationSCP1',
    'BasicMotions', 'Epilepsy', 'Heartbeat', 'NATOPS', 'SelfRegulationSCP2',
    'CharacterTrajectories', 'EthanolConcentration', 'InsectWingbeat', 'PEMS-SF', 'SpokenArabicDigits',
    'Cricket', 'FaceDetection', 'JapaneseVowels', 'PenDigits', 'StandWalkJump',
    'DuckDuckGeese', 'FingerMovements', 'LSST', 'PhonemeSpectra', 'UWaveGestureLibrary'
]

"""List of non-normalized multivariate datasets from UEA archive"""
multivar_non_normalized_datasets = [
    'AtrialFibrillation',
    'BasicMotions',
    'CharacterTrajectories',
    'DuckDuckGeese',
    'EigenWorms',
    'Epilepsy',
    'EthanolConcentration',
    'ERing',
    'FaceDetection',
    'FingerMovements',
    'HandMovementDirection',
    'Heartbeat',
    'InsectWingbeat',
    'JapaneseVowels',
    'Libras',
    'LSST',
    'MotorImagery',
    'NATOPS',
    'PenDigits',
    'PEMS-SF',
    'Phoneme',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'StandWalkJump'
]
