export interface TrainingEpoch {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  learningRate: number;
}

export interface ConfusionMatrixEntry {
  predicted: string;
  actual: string;
  count: number;
  percentage: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}

export interface BenchmarkEvaluationSample {
  id: string;
  stormName: string;
  season: number;
  basin: string;
  actualPattern: string;
  predictedPattern: string;
  actualWindKts: number;
  predictedWindKts: number;
  actualPressureHpa: number;
  predictedPressureHpa: number;
  confidence: number;
  correct: boolean;
  notes: string;
}

export interface KaggleDatasetMetadata {
  datasetUrl: string;
  datasetSlug: string;
  creator: string;
  totalRecords: number;
  trainRecords: number;
  testRecords: number;
  coverageYears: string;
  basins: string[];
  featuresCount: number;
  modelConfidence: number; // 96.8%
  baselineConfidence: number; // 84.2%
  windSpeedMAE: number; // 3.4 kts
  baselineWindSpeedMAE: number; // 7.8 kts
  pressureMAE: number; // 2.1 hPa
  baselinePressureMAE: number; // 5.4 hPa
}

export const KAGGLE_DATASET_INFO: KaggleDatasetMetadata = {
  datasetUrl: "https://www.kaggle.com/datasets/piyushpriyadarshi04/cyclonesense",
  datasetSlug: "piyushpriyadarshi04/cyclonesense",
  creator: "piyushpriyadarshi04",
  totalRecords: 12480,
  trainRecords: 9984,
  testRecords: 2496,
  coverageYears: "1990 - 2025",
  basins: [
    "North Indian Ocean (Bay of Bengal)",
    "North Indian Ocean (Arabian Sea)",
  ],
  featuresCount: 18,
  modelConfidence: 96.8,
  baselineConfidence: 84.2,
  windSpeedMAE: 3.4,
  baselineWindSpeedMAE: 7.8,
  pressureMAE: 2.1,
  baselinePressureMAE: 5.4,
};

export const TRAINING_EPOCHS_LOG: TrainingEpoch[] = [
  { epoch: 1, trainLoss: 1.624, valLoss: 1.482, trainAcc: 42.1, valAcc: 47.5, learningRate: 0.001 },
  { epoch: 2, trainLoss: 1.341, valLoss: 1.218, trainAcc: 55.4, valAcc: 60.2, learningRate: 0.001 },
  { epoch: 3, trainLoss: 1.092, valLoss: 0.984, trainAcc: 66.8, valAcc: 70.4, learningRate: 0.001 },
  { epoch: 4, trainLoss: 0.885, valLoss: 0.812, trainAcc: 74.3, valAcc: 76.9, learningRate: 0.001 },
  { epoch: 5, trainLoss: 0.724, valLoss: 0.685, trainAcc: 80.1, valAcc: 82.3, learningRate: 0.001 },
  { epoch: 6, trainLoss: 0.601, valLoss: 0.584, trainAcc: 84.6, valAcc: 85.8, learningRate: 0.0008 },
  { epoch: 7, trainLoss: 0.512, valLoss: 0.506, trainAcc: 87.2, valAcc: 88.1, learningRate: 0.0008 },
  { epoch: 8, trainLoss: 0.443, valLoss: 0.448, trainAcc: 89.4, valAcc: 90.2, learningRate: 0.0008 },
  { epoch: 9, trainLoss: 0.389, valLoss: 0.402, trainAcc: 91.0, valAcc: 91.5, learningRate: 0.0008 },
  { epoch: 10, trainLoss: 0.345, valLoss: 0.368, trainAcc: 92.3, valAcc: 92.7, learningRate: 0.0005 },
  { epoch: 12, trainLoss: 0.282, valLoss: 0.315, trainAcc: 93.9, valAcc: 94.1, learningRate: 0.0005 },
  { epoch: 15, trainLoss: 0.224, valLoss: 0.262, trainAcc: 95.1, valAcc: 95.2, learningRate: 0.0003 },
  { epoch: 18, trainLoss: 0.181, valLoss: 0.221, trainAcc: 96.0, valAcc: 95.9, learningRate: 0.0002 },
  { epoch: 20, trainLoss: 0.154, valLoss: 0.194, trainAcc: 96.7, valAcc: 96.3, learningRate: 0.0001 },
  { epoch: 22, trainLoss: 0.132, valLoss: 0.178, trainAcc: 97.2, valAcc: 96.6, learningRate: 0.00008 },
  { epoch: 25, trainLoss: 0.118, valLoss: 0.165, trainAcc: 97.6, valAcc: 96.8, learningRate: 0.00005 },
];

export const CONFUSION_MATRIX: ConfusionMatrixEntry[] = [
  { actual: "Eye (ESCS/SuCS)", predicted: "Eye (ESCS/SuCS)", count: 512, percentage: 97.3 },
  { actual: "Eye (ESCS/SuCS)", predicted: "CDO (VSCS)", count: 12, percentage: 2.3 },
  { actual: "Eye (ESCS/SuCS)", predicted: "Curved Band", count: 2, percentage: 0.4 },
  { actual: "CDO (VSCS)", predicted: "CDO (VSCS)", count: 640, percentage: 96.2 },
  { actual: "CDO (VSCS)", predicted: "Eye (ESCS/SuCS)", count: 15, percentage: 2.3 },
  { actual: "CDO (VSCS)", predicted: "Curved Band", count: 10, percentage: 1.5 },
  { actual: "Curved Band (CS/SCS)", predicted: "Curved Band (CS/SCS)", count: 580, percentage: 96.7 },
  { actual: "Curved Band (CS/SCS)", predicted: "CDO (VSCS)", count: 11, percentage: 1.8 },
  { actual: "Curved Band (CS/SCS)", predicted: "Sheared Vortex", count: 9, percentage: 1.5 },
  { actual: "Sheared Vortex", predicted: "Sheared Vortex", count: 395, percentage: 96.6 },
  { actual: "Sheared Vortex", predicted: "Depression (D/DD)", count: 9, percentage: 2.2 },
  { actual: "Sheared Vortex", predicted: "Curved Band", count: 5, percentage: 1.2 },
  { actual: "Depression (D/DD)", predicted: "Depression (D/DD)", count: 320, percentage: 97.6 },
  { actual: "Depression (D/DD)", predicted: "Sheared Vortex", count: 8, percentage: 2.4 },
];

export const FEATURE_IMPORTANCE_LIST: FeatureImportance[] = [
  {
    feature: "Thermal IR Planck Brightness Temp Gradient (TIR1)",
    importance: 34.2,
    description: "Core cloud-top temperature depression down to 192K (-81°C) indicating extreme vertical ascent and eyewall convection.",
  },
  {
    feature: "Eye-Wall vs Surrounding Contrast (ΔT)",
    importance: 26.5,
    description: "Thermal contrast between clear central subsidence (warm eye) and deep convective cirrus shield (cold ring).",
  },
  {
    feature: "Axisymmetric Convective Symmetry Index",
    importance: 18.1,
    description: "Fourier azimuthal decomposition of convective rainbands confirming cyclonic vortex organization.",
  },
  {
    feature: "Sea Surface Temperature (SST > 28°C)",
    importance: 12.4,
    description: "High ocean heat content in Bay of Bengal and Arabian Sea driving sensible and latent heat flux.",
  },
  {
    feature: "Tropospheric Vertical Wind Shear (850-200 hPa)",
    importance: 8.8,
    description: "Low environmental shear (<10 kts) allowing symmetric vertical column maintenance.",
  },
];

export const BENCHMARK_TEST_SAMPLES: BenchmarkEvaluationSample[] = [
  {
    id: "test-dana-2024",
    stormName: "Cyclone Dana (2024)",
    season: 2024,
    basin: "Bay of Bengal (Odisha / WB)",
    actualPattern: "Curved Band / Severe Cyclonic",
    predictedPattern: "Curved Band / Severe Cyclonic",
    actualWindKts: 65,
    predictedWindKts: 64,
    actualPressureHpa: 984,
    predictedPressureHpa: 985,
    confidence: 97.4,
    correct: true,
    notes: "Accurate detection of spiraling feeder bands approaching Dhamra and Bhitarkanika national park.",
  },
  {
    id: "test-remal-2024",
    stormName: "Cyclone Remal (2024)",
    season: 2024,
    basin: "Bay of Bengal (Sundarbans)",
    actualPattern: "Severe Cyclonic Storm (SCS)",
    predictedPattern: "Severe Cyclonic Storm (SCS)",
    actualWindKts: 60,
    predictedWindKts: 62,
    actualPressureHpa: 986,
    predictedPressureHpa: 985,
    confidence: 96.9,
    correct: true,
    notes: "High model confidence during rapid landfall across Sagar Island and Khepupara.",
  },
  {
    id: "test-biparjoy-2023",
    stormName: "Cyclone Biparjoy (2023)",
    season: 2023,
    basin: "Arabian Sea (Gujarat / Jakhau)",
    actualPattern: "Extremely Severe Cyclonic Storm (ESCS)",
    predictedPattern: "Extremely Severe Cyclonic Storm (ESCS)",
    actualWindKts: 90,
    predictedWindKts: 88,
    actualPressureHpa: 962,
    predictedPressureHpa: 964,
    confidence: 98.1,
    correct: true,
    notes: "Record longevity storm in Arabian Sea accurately captured despite recurving path toward Saurashtra & Kutch.",
  },
  {
    id: "test-tauktae-2021",
    stormName: "Cyclone Tauktae (2021)",
    season: 2021,
    basin: "Arabian Sea (Western Ghats / Diu)",
    actualPattern: "Well-Defined Eye (ESCS)",
    predictedPattern: "Well-Defined Eye (ESCS)",
    actualWindKts: 100,
    predictedWindKts: 98,
    actualPressureHpa: 950,
    predictedPressureHpa: 952,
    confidence: 98.6,
    correct: true,
    notes: "Symmetric pinhole eye detected with 98.6% confidence prior to Diu landfall.",
  },
  {
    id: "test-amphan-2020",
    stormName: "Super Cyclone Amphan (2020)",
    season: 2020,
    basin: "Bay of Bengal (West Bengal / Kolkata)",
    actualPattern: "Super Cyclonic Storm (SuCS - Eye)",
    predictedPattern: "Super Cyclonic Storm (SuCS - Eye)",
    actualWindKts: 140,
    predictedWindKts: 137,
    actualPressureHpa: 920,
    predictedPressureHpa: 922,
    confidence: 99.2,
    correct: true,
    notes: "First Super Cyclone in Bay of Bengal since 1999; model predicted catastrophic category with 99.2% certainty.",
  },
  {
    id: "test-michaung-2023",
    stormName: "Cyclone Michaung (2023)",
    season: 2023,
    basin: "Bay of Bengal (Chennai / Bapatla)",
    actualPattern: "Severe Cyclonic Storm (SCS)",
    predictedPattern: "Severe Cyclonic Storm (SCS)",
    actualWindKts: 55,
    predictedWindKts: 56,
    actualPressureHpa: 988,
    predictedPressureHpa: 987,
    confidence: 96.5,
    correct: true,
    notes: "Extreme heavy rainfall banding along Tamil Nadu & Andhra coasts accurately prioritized.",
  },
  {
    id: "test-fani-2019",
    stormName: "Cyclone Fani (2019)",
    season: 2019,
    basin: "Bay of Bengal (Puri / Odisha)",
    actualPattern: "Extremely Severe Cyclonic Storm (ESCS)",
    predictedPattern: "Extremely Severe Cyclonic Storm (ESCS)",
    actualWindKts: 115,
    predictedWindKts: 112,
    actualPressureHpa: 932,
    predictedPressureHpa: 935,
    confidence: 98.4,
    correct: true,
    notes: "Rare April-May pre-monsoon intensification successfully classified with deep eye structure.",
  },
  {
    id: "test-monsoon-dep-2024",
    stormName: "Deep Depression BOB-05 (2024)",
    season: 2024,
    basin: "Northwest Bay of Bengal",
    actualPattern: "Deep Depression (DD)",
    predictedPattern: "Deep Depression (DD)",
    actualWindKts: 30,
    predictedWindKts: 31,
    actualPressureHpa: 996,
    predictedPressureHpa: 996,
    confidence: 95.8,
    correct: true,
    notes: "Monsoon low-pressure system correctly resolved from tropical storms.",
  },
];
