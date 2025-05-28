import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileWriter;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * Improved Genetic Programming Classifier for Financial Stock Prediction
 * COS314 Assignment 3
 *
 * This implementation addresses data leakage issues and provides better
 * validation for time-series financial data classification.
 */
public class ImprovedGeneticProgrammingClassifier {

    // GP Parameters - adjusted for better performance
    private static final int POPULATION_SIZE = 100;
    private static final int MAX_GENERATIONS = 50;
    private static final double CROSSOVER_RATE = 0.7;
    private static final double INITIAL_MUTATION_RATE = 0.15;
    private static final int TOURNAMENT_SIZE = 7;
    private static final int MAX_TREE_DEPTH = 4;
    private static final int ELITE_SIZE = 2; // Number of best individuals to preserve
    private static final int EARLY_STOPPING_PATIENCE = 10; // Generations without improvement before stopping
    private static final double MIN_MUTATION_RATE = 0.05;
    private static final double MAX_MUTATION_RATE = 0.3;
    
    // Data parameters
    private static final int NUM_FEATURES = 5; // Open, High, Low, Close, Adj Close
    
    // Random generator - will be seeded
    private static Random random;
    
    // Training dataset
    private static List<double[]> trainFeatures = new ArrayList<>();
    private static List<Integer> trainLabels = new ArrayList<>();
    private static List<String> trainDates = new ArrayList<>();
    
    // Test dataset
    private static List<double[]> testFeatures = new ArrayList<>();
    private static List<Integer> testLabels = new ArrayList<>();
    private static List<String> testDates = new ArrayList<>();
    
    // Validation dataset (split from training for better evaluation)
    private static List<double[]> validationFeatures = new ArrayList<>();
    private static List<Integer> validationLabels = new ArrayList<>();
    
    // Feature scaling parameters
    private static double[] featureMeans;
    private static double[] featureStdDevs;
    
    // Early stopping variables
    private static double bestValidationFitness = Double.NEGATIVE_INFINITY;
    private static int generationsWithoutImprovement = 0;
    
    // Thread pool for parallel evaluation
    private static ExecutorService executorService;

    // Add result tracking variables
    private static long currentSeed;
    private static final String RESULTS_FILE = "gp_results.csv";
    private static boolean resultsFileExists = false;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        // Initialize results file if it doesn't exist
        initializeResultsFile();
        
        // Prompt for seed
        System.out.print("Enter a seed value (press Enter for random seed): ");
        String seedInput = scanner.nextLine();
        
        if (seedInput.trim().isEmpty()) {
            currentSeed = System.currentTimeMillis();
            System.out.println("Using generated seed: " + currentSeed);
        } else {
            try {
                currentSeed = Long.parseLong(seedInput);
            } catch (NumberFormatException e) {
                currentSeed = seedInput.hashCode();
                System.out.println("Converting non-numeric input to seed: " + currentSeed);
            }
        }
        
        random = new Random(currentSeed);
        
        // Load training data
        String trainFilePath;
        do {
            System.out.print("Enter the path to the training CSV data file: ");
            trainFilePath = scanner.nextLine();
            if (trainFilePath.trim().isEmpty()) {
                System.out.println("File path cannot be empty. Please try again.");
            }
        } while (trainFilePath.trim().isEmpty());
        
        loadDataWithDateValidation(trainFilePath, trainFeatures, trainLabels, trainDates, "training");
        
        if (trainFeatures.isEmpty()) {
            System.out.println("No training data loaded. Exiting.");
            return;
        }
        
        // Load test data
        String testFilePath;
        do {
            System.out.print("Enter the path to the test CSV data file: ");
            testFilePath = scanner.nextLine();
            if (testFilePath.trim().isEmpty()) {
                System.out.println("File path cannot be empty. Please try again.");
            }
        } while (testFilePath.trim().isEmpty());
        
        loadDataWithDateValidation(testFilePath, testFeatures, testLabels, testDates, "test");
        
        if (testFeatures.isEmpty()) {
            System.out.println("No test data loaded. Exiting.");
            return;
        }
        
        // Validate chronological order
        if (!validateChronologicalOrder()) {
            System.out.println("WARNING: Potential data leakage detected!");
            System.out.println("Please ensure training data comes chronologically before test data.");
            return;
        }
        
        // Create validation split from training data (20% of training data)
        createValidationSplit();
        
        System.out.println("Training data loaded successfully. Records: " + trainFeatures.size());
        System.out.println("Validation data created. Records: " + validationFeatures.size());
        System.out.println("Test data loaded successfully. Records: " + testFeatures.size());
        
        // Run the genetic programming algorithm
        TreeNode bestModel = runGeneticProgramming();
        
        // Comprehensive evaluation
        System.out.println("\n" + "=".repeat(60));
        System.out.println("FINAL MODEL EVALUATION");
        System.out.println("=".repeat(60));
        
        System.out.println("\nTRAINING SET EVALUATION");
        System.out.println("-".repeat(30));
        double trainAccuracy = evaluateModel(bestModel, trainFeatures, trainLabels);
        
        System.out.println("\nVALIDATION SET EVALUATION");
        System.out.println("-".repeat(30));
        double validationAccuracy = evaluateModel(bestModel, validationFeatures, validationLabels);
        
        System.out.println("\nTEST SET EVALUATION");
        System.out.println("-".repeat(30));
        double testAccuracy = evaluateModel(bestModel, testFeatures, testLabels);
        
        // Analysis and warnings
        analyzeResults(trainAccuracy, validationAccuracy, testAccuracy);
        
        // After all evaluations, save results
        saveResults(trainAccuracy, validationAccuracy, testAccuracy, bestModel);
        
        scanner.close();
    }
    
    /**
     * Enhanced data loading with date validation
     */
    private static void loadDataWithDateValidation(String filePath, List<double[]> features, 
                                                 List<Integer> labels, List<String> dates, String datasetType) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine(); // Read header
            System.out.println("Header for " + datasetType + " data: " + line);
            
            // Determine if date column exists
            String[] headerColumns = line.split(",");
            boolean hasDateColumn = headerColumns.length > NUM_FEATURES + 1;
            int dateColumnIndex = hasDateColumn ? 0 : -1;
            int featureStartIndex = hasDateColumn ? 1 : 0;
            int labelIndex = hasDateColumn ? NUM_FEATURES + 1 : NUM_FEATURES;
            
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length >= (hasDateColumn ? NUM_FEATURES + 2 : NUM_FEATURES + 1)) {
                    // Extract date if available
                    if (hasDateColumn) {
                        dates.add(values[dateColumnIndex].trim());
                    } else {
                        dates.add("N/A");
                    }
                    
                    // Extract features
                    double[] featureVector = new double[NUM_FEATURES];
                    for (int i = 0; i < NUM_FEATURES; i++) {
                        featureVector[i] = Double.parseDouble(values[featureStartIndex + i]);
                    }
                    
                    // Extract label
                    int label = Integer.parseInt(values[labelIndex].trim());
                    
                    features.add(featureVector);
                    labels.add(label);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading " + datasetType + " file: " + e.getMessage());
        } catch (NumberFormatException e) {
            System.err.println("Error parsing number in " + datasetType + " file: " + e.getMessage());
        }
    }
    
    /**
     * Validate that training data comes before test data chronologically
     */
    private static boolean validateChronologicalOrder() {
        if (trainDates.isEmpty() || testDates.isEmpty() || 
            trainDates.get(0).equals("N/A") || testDates.get(0).equals("N/A")) {
            System.out.println("Date information not available - cannot validate chronological order.");
            System.out.println("Please ensure your data is properly ordered chronologically.");
            return true; // Continue but warn
        }
        
        try {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
            
            // Get the latest date from training data
            Date latestTrainDate = null;
            for (String dateStr : trainDates) {
                Date date = dateFormat.parse(dateStr);
                if (latestTrainDate == null || date.after(latestTrainDate)) {
                    latestTrainDate = date;
                }
            }
            
            // Get the earliest date from test data
            Date earliestTestDate = null;
            for (String dateStr : testDates) {
                Date date = dateFormat.parse(dateStr);
                if (earliestTestDate == null || date.before(earliestTestDate)) {
                    earliestTestDate = date;
                }
            }
            
            if (latestTrainDate != null && earliestTestDate != null) {
                System.out.println("Latest training date: " + dateFormat.format(latestTrainDate));
                System.out.println("Earliest test date: " + dateFormat.format(earliestTestDate));
                
                if (latestTrainDate.after(earliestTestDate) || latestTrainDate.equals(earliestTestDate)) {
                    System.out.println("ERROR: Training data overlaps with or comes after test data!");
                    return false;
                }
            }
            
            return true;
        } catch (ParseException e) {
            System.out.println("Could not parse dates - assuming chronological order is correct.");
            return true;
        }
    }
    
    /**
     * Create validation split from training data
     */
    private static void createValidationSplit() {
        int validationSize = (int)(trainFeatures.size() * 0.2); // 20% for validation
        
        // Always take the last 20% for validation to maintain chronological order
        validationFeatures = new ArrayList<>(trainFeatures.subList(trainFeatures.size() - validationSize, trainFeatures.size()));
        validationLabels = new ArrayList<>(trainLabels.subList(trainLabels.size() - validationSize, trainLabels.size()));
        
        // Update training set to exclude validation data
        trainFeatures = new ArrayList<>(trainFeatures.subList(0, trainFeatures.size() - validationSize));
        trainLabels = new ArrayList<>(trainLabels.subList(0, trainLabels.size() - validationSize));
    }
    
    /**
     * Enhanced genetic programming with validation-based early stopping
     */
    private static TreeNode runGeneticProgramming() {
        // Initialize thread pool
        executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        
        // Scale features
        scaleFeatures();
        
        // Initialize population
        List<TreeNode> population = initializePopulation();
        List<TreeNode> elite = new ArrayList<>();
        double currentMutationRate = INITIAL_MUTATION_RATE;
        
        for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
            // Evaluate population in parallel
            Map<TreeNode, Double> fitnessScores = evaluatePopulationParallel(population);
            
            // Update best validation fitness and check early stopping
            double currentBestFitness = Collections.max(fitnessScores.values());
            if (currentBestFitness > bestValidationFitness) {
                bestValidationFitness = currentBestFitness;
                generationsWithoutImprovement = 0;
            } else {
                generationsWithoutImprovement++;
            }
            
            if (generationsWithoutImprovement >= EARLY_STOPPING_PATIENCE) {
                System.out.println("Early stopping triggered at generation " + generation);
                break;
            }
            
            // Select elite individuals
            elite.clear();
            fitnessScores.entrySet().stream()
                .sorted(Map.Entry.<TreeNode, Double>comparingByValue().reversed())
                .limit(ELITE_SIZE)
                .forEach(entry -> elite.add(entry.getKey().deepCopy()));
            
            // Create new population
            List<TreeNode> newPopulation = new ArrayList<>(elite);
            
            // Adaptive mutation rate
            currentMutationRate = Math.max(MIN_MUTATION_RATE,
                Math.min(MAX_MUTATION_RATE,
                    INITIAL_MUTATION_RATE * (1.0 - (double)generation / MAX_GENERATIONS)));
            
            while (newPopulation.size() < POPULATION_SIZE) {
                TreeNode parent1 = tournamentSelection(population, fitnessScores);
                TreeNode parent2 = tournamentSelection(population, fitnessScores);
                
                if (random.nextDouble() < CROSSOVER_RATE) {
                    TreeNode[] children = crossover(parent1, parent2);
                    newPopulation.add(children[0]);
                    if (newPopulation.size() < POPULATION_SIZE) {
                        newPopulation.add(children[1]);
                    }
                } else {
                    newPopulation.add(parent1.deepCopy());
                    if (newPopulation.size() < POPULATION_SIZE) {
                        newPopulation.add(parent2.deepCopy());
                    }
                }
            }
            
            // Apply mutation
            for (int i = ELITE_SIZE; i < newPopulation.size(); i++) {
                if (random.nextDouble() < currentMutationRate) {
                    mutate(newPopulation.get(i));
                }
            }
            
            population = newPopulation;
            
            // Print progress
            if (generation % 5 == 0) {
                System.out.printf("Generation %d: Best Fitness = %.4f, Mutation Rate = %.4f%n",
                    generation, currentBestFitness, currentMutationRate);
            }
        }
        
        // Shutdown thread pool
        executorService.shutdown();
        
        // Return best individual
        return elite.get(0);
    }
    
    private static Map<TreeNode, Double> evaluatePopulationParallel(List<TreeNode> population) {
        // Use LinkedHashMap to preserve insertion order
        Map<TreeNode, Double> fitnessScores = new LinkedHashMap<>();
        
        // Evaluate in deterministic order
        for (TreeNode individual : population) {
            double fitness = calculateValidationFitness(individual);
            fitnessScores.put(individual, fitness);
        }
        
        return fitnessScores;
    }
    
    private static double calculateValidationFitness(TreeNode individual) {
        double accuracy = evaluateModel(individual, validationFeatures, validationLabels);
        double complexity = getTreeSize(individual) / 100.0; // Normalize complexity penalty
        return accuracy - 0.1 * complexity; // Add complexity penalty to prevent overfitting
    }
    
    /**
     * Initialize population with diversity control
     */
    private static List<TreeNode> initializePopulation() {
        List<TreeNode> population = new ArrayList<>();
        
        for (int i = 0; i < POPULATION_SIZE; i++) {
            TreeNode individual = generateRandomTree(0, MAX_TREE_DEPTH);
            population.add(individual);
        }
        
        return population;
    }
    
    /**
     * Generate random tree with improved diversity
     */
    private static TreeNode generateRandomTree(int currentDepth, int maxDepth) {
        // Increase probability of terminals at deeper levels
        double terminalProbability = currentDepth >= maxDepth ? 1.0 : 
                                   Math.min(0.6, 0.3 + (double) currentDepth / maxDepth * 0.3);
        
        if (currentDepth >= maxDepth || random.nextDouble() < terminalProbability) {
            if (random.nextBoolean()) {
                int featureIndex = random.nextInt(NUM_FEATURES);
                return new FeatureNode(featureIndex);
            } else {
                // Use nextDouble instead of nextGaussian for more stable random values
                double value = (random.nextDouble() * 2 - 1) * 0.5; // Range: [-0.5, 0.5]
                return new ConstantNode(value);
            }
        } else {
            // Use consistent ordering for node types
            int nodeType = random.nextInt(5);
            TreeNode left = generateRandomTree(currentDepth + 1, maxDepth);
            TreeNode right = generateRandomTree(currentDepth + 1, maxDepth);
            TreeNode extra = nodeType == 4 ? generateRandomTree(currentDepth + 1, maxDepth) : null;
            
            switch (nodeType) {
                case 0: return new AddNode(left, right);
                case 1: return new SubtractNode(left, right);
                case 2: return new MultiplyNode(left, right);
                case 3: return new SafeDivideNode(left, right);
                case 4: return new IfNode(left, right, extra);
                default: return new ConstantNode(0.0);
            }
        }
    }
    
    /**
     * Calculate tree size for complexity penalty
     */
    private static int getTreeSize(TreeNode node) {
        if (node instanceof BinaryNode) {
            BinaryNode bNode = (BinaryNode) node;
            return 1 + getTreeSize(bNode.left) + getTreeSize(bNode.right);
        } else if (node instanceof IfNode) {
            IfNode ifNode = (IfNode) node;
            return 1 + getTreeSize(ifNode.condition) + getTreeSize(ifNode.ifTrue) + getTreeSize(ifNode.ifFalse);
        } else {
            return 1;
        }
    }
    
    /**
     * Enhanced crossover with better subtree selection
     */
    private static TreeNode[] crossover(TreeNode parent1, TreeNode parent2) {
        TreeNode offspring1 = parent1.deepCopy();
        TreeNode offspring2 = parent2.deepCopy();
        
        List<TreeNode> nodes1 = getAllNodes(offspring1);
        List<TreeNode> nodes2 = getAllNodes(offspring2);
        
        if (nodes1.size() <= 1 || nodes2.size() <= 1) {
            return new TreeNode[] { offspring1, offspring2 };
        }
        
        // Prefer internal nodes for crossover (90% of the time)
        int point1 = random.nextDouble() < 0.9 && nodes1.size() > 2 ? 
                    1 + random.nextInt(Math.min(nodes1.size() - 1, nodes1.size() / 2)) :
                    1 + random.nextInt(nodes1.size() - 1);
        int point2 = random.nextDouble() < 0.9 && nodes2.size() > 2 ? 
                    1 + random.nextInt(Math.min(nodes2.size() - 1, nodes2.size() / 2)) :
                    1 + random.nextInt(nodes2.size() - 1);
        
        TreeNode subtree1 = nodes1.get(point1);
        TreeNode subtree2 = nodes2.get(point2);
        
        TreeNode parent1Node = findParent(offspring1, subtree1);
        TreeNode parent2Node = findParent(offspring2, subtree2);
        
        if (parent1Node != null && parent2Node != null) {
            replaceChild(parent1Node, subtree1, subtree2.deepCopy());
            replaceChild(parent2Node, subtree2, subtree1.deepCopy());
        }
        
        return new TreeNode[] { offspring1, offspring2 };
    }
    
    /**
     * Helper method to replace a child in a parent node
     */
    private static void replaceChild(TreeNode parent, TreeNode oldChild, TreeNode newChild) {
        if (parent instanceof BinaryNode) {
            BinaryNode bNode = (BinaryNode) parent;
            if (bNode.left == oldChild) {
                bNode.left = newChild;
            } else if (bNode.right == oldChild) {
                bNode.right = newChild;
            }
        } else if (parent instanceof IfNode) {
            IfNode ifNode = (IfNode) parent;
            if (ifNode.condition == oldChild) {
                ifNode.condition = newChild;
            } else if (ifNode.ifTrue == oldChild) {
                ifNode.ifTrue = newChild;
            } else if (ifNode.ifFalse == oldChild) {
                ifNode.ifFalse = newChild;
            }
        }
    }
    
    /**
     * Find parent node of target
     */
    private static TreeNode findParent(TreeNode root, TreeNode target) {
        if (root instanceof BinaryNode) {
            BinaryNode bNode = (BinaryNode) root;
            if (bNode.left == target || bNode.right == target) {
                return root;
            }
            TreeNode leftResult = findParent(bNode.left, target);
            if (leftResult != null) return leftResult;
            return findParent(bNode.right, target);
        } else if (root instanceof IfNode) {
            IfNode ifNode = (IfNode) root;
            if (ifNode.condition == target || ifNode.ifTrue == target || ifNode.ifFalse == target) {
                return root;
            }
            TreeNode condResult = findParent(ifNode.condition, target);
            if (condResult != null) return condResult;
            TreeNode trueResult = findParent(ifNode.ifTrue, target);
            if (trueResult != null) return trueResult;
            return findParent(ifNode.ifFalse, target);
        }
        return null;
    }
    
    /**
     * Get all nodes in tree
     */
    private static List<TreeNode> getAllNodes(TreeNode root) {
        List<TreeNode> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        return nodes;
    }
    
    /**
     * Collect nodes recursively
     */
    private static void collectNodes(TreeNode node, List<TreeNode> nodes) {
        nodes.add(node);
        
        if (node instanceof BinaryNode) {
            BinaryNode bNode = (BinaryNode) node;
            collectNodes(bNode.left, nodes);
            collectNodes(bNode.right, nodes);
        } else if (node instanceof IfNode) {
            IfNode ifNode = (IfNode) node;
            collectNodes(ifNode.condition, nodes);
            collectNodes(ifNode.ifTrue, nodes);
            collectNodes(ifNode.ifFalse, nodes);
        }
    }
    
    /**
     * Enhanced mutation with multiple strategies
     */
    private static void mutate(TreeNode tree) {
        List<TreeNode> nodes = getAllNodes(tree);
        
        if (nodes.size() <= 1) return;
        
        int mutationType = random.nextInt(3);
        
        switch (mutationType) {
            case 0: // Point mutation - change a terminal
                mutateTerminal(tree, nodes);
                break;
            case 1: // Subtree mutation - replace subtree
                mutateSubtree(tree, nodes);
                break;
            case 2: // Grow mutation - extend a terminal
                growMutation(tree, nodes);
                break;
        }
    }
    
    private static void mutateTerminal(TreeNode tree, List<TreeNode> nodes) {
        List<TreeNode> terminals = new ArrayList<>();
        for (TreeNode node : nodes) {
            if (node instanceof ConstantNode || node instanceof FeatureNode) {
                terminals.add(node);
            }
        }
        
        if (terminals.isEmpty()) return;
        
        TreeNode terminal = terminals.get(random.nextInt(terminals.size()));
        TreeNode parent = findParent(tree, terminal);
        
        if (parent != null) {
            TreeNode newTerminal;
            if (random.nextBoolean()) {
                newTerminal = new FeatureNode(random.nextInt(NUM_FEATURES));
            } else {
                newTerminal = new ConstantNode(random.nextGaussian() * 0.5);
            }
            replaceChild(parent, terminal, newTerminal);
        }
    }
    
    private static void mutateSubtree(TreeNode tree, List<TreeNode> nodes) {
        if (nodes.size() <= 1) return;
        
        int nodeIndex = 1 + random.nextInt(nodes.size() - 1);
        TreeNode nodeToMutate = nodes.get(nodeIndex);
        TreeNode parent = findParent(tree, nodeToMutate);
        
        if (parent != null) {
            TreeNode newSubtree = generateRandomTree(0, MAX_TREE_DEPTH / 2);
            replaceChild(parent, nodeToMutate, newSubtree);
        }
    }
    
    private static void growMutation(TreeNode tree, List<TreeNode> nodes) {
        List<TreeNode> terminals = new ArrayList<>();
        for (TreeNode node : nodes) {
            if (node instanceof ConstantNode || node instanceof FeatureNode) {
                terminals.add(node);
            }
        }
        
        if (terminals.isEmpty()) return;
        
        TreeNode terminal = terminals.get(random.nextInt(terminals.size()));
        TreeNode parent = findParent(tree, terminal);
        
        if (parent != null && getDepth(tree, terminal) < MAX_TREE_DEPTH - 1) {
            TreeNode newFunction = generateRandomTree(0, 2);
            replaceChild(parent, terminal, newFunction);
        }
    }
    
    private static int getDepth(TreeNode root, TreeNode target) {
        return getDepthHelper(root, target, 0);
    }
    
    private static int getDepthHelper(TreeNode current, TreeNode target, int currentDepth) {
        if (current == target) return currentDepth;
        
        if (current instanceof BinaryNode) {
            BinaryNode bNode = (BinaryNode) current;
            int leftDepth = getDepthHelper(bNode.left, target, currentDepth + 1);
            if (leftDepth != -1) return leftDepth;
            return getDepthHelper(bNode.right, target, currentDepth + 1);
        } else if (current instanceof IfNode) {
            IfNode ifNode = (IfNode) current;
            int condDepth = getDepthHelper(ifNode.condition, target, currentDepth + 1);
            if (condDepth != -1) return condDepth;
            int trueDepth = getDepthHelper(ifNode.ifTrue, target, currentDepth + 1);
            if (trueDepth != -1) return trueDepth;
            return getDepthHelper(ifNode.ifFalse, target, currentDepth + 1);
        }
        
        return -1;
    }
    
   /**
     * Enhanced model evaluation with detailed metrics
     */
    private static double evaluateModel(TreeNode model, List<double[]> features, List<Integer> labels) {
        int truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
        
        for (int i = 0; i < features.size(); i++) {
            double[] input = features.get(i);
            int actualClass = labels.get(i);
            
            double output = model.evaluate(input);
            int predictedClass = output >= 0 ? 1 : 0;
            
            if (predictedClass == 1 && actualClass == 1) truePositives++;
            else if (predictedClass == 1 && actualClass == 0) falsePositives++;
            else if (predictedClass == 0 && actualClass == 0) trueNegatives++;
            else falseNegatives++;
        }
        
        double accuracy = (double) (truePositives + trueNegatives) / (truePositives + falsePositives + trueNegatives + falseNegatives);
        double precision = truePositives == 0 ? 0 : (double) truePositives / (truePositives + falsePositives);
        double recall = truePositives == 0 ? 0 : (double) truePositives / (truePositives + falseNegatives);
        double f1Score = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
        
        System.out.println("Accuracy:  " + String.format("%.4f", accuracy));
        System.out.println("Precision: " + String.format("%.4f", precision));
        System.out.println("Recall:    " + String.format("%.4f", recall));
        System.out.println("F1-Score:  " + String.format("%.4f", f1Score));
        System.out.println("True Positives:  " + truePositives);
        System.out.println("False Positives: " + falsePositives);
        System.out.println("True Negatives:  " + trueNegatives);
        System.out.println("False Negatives: " + falseNegatives);
        
        return accuracy; // Return accuracy for general comparison, F1 is used for fitness
    }
    
    /**
     * Analyze results and provide warnings for potential issues like overfitting or data leakage
     */
    private static void analyzeResults(double trainAccuracy, double validationAccuracy, double testAccuracy) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("RESULT ANALYSIS AND WARNINGS");
        System.out.println("=".repeat(60));
        
        if (trainAccuracy > validationAccuracy * 1.1) { // If train accuracy is significantly higher
            System.out.println("WARNING: Potential Overfitting Detected!");
            System.out.println("Training accuracy (" + String.format("%.4f", trainAccuracy) + 
                               ") is much higher than validation accuracy (" + String.format("%.4f", validationAccuracy) + ").");
            System.out.println("Consider: Increasing regularization, reducing MAX_TREE_DEPTH, or increasing population diversity.");
        }
        
        if (validationAccuracy < testAccuracy && testAccuracy > trainAccuracy * 1.05) { // Test accuracy significantly higher than validation and training
            System.out.println("CRITICAL WARNING: Test Accuracy is significantly higher than Training/Validation Accuracy!");
            System.out.println("This is a strong indicator of DATA LEAKAGE or a statistical anomaly due to a small test set.");
            System.out.println("Training Accuracy: " + String.format("%.4f", trainAccuracy));
            System.out.println("Validation Accuracy: " + String.format("%.4f", validationAccuracy));
            System.out.println("Test Accuracy: " + String.format("%.4f", testAccuracy));
            System.out.println("Action Required: Re-verify chronological split of your data (training must strictly precede test).");
            System.out.println("Ensure no random shuffling of time-series data occurred before splitting.");
            System.out.println("If test set is very small, this could also be a statistical fluke.");
        } else if (testAccuracy < validationAccuracy * 0.9) { // If test accuracy is significantly lower than validation
            System.out.println("WARNING: Model generalization might be poor on unseen data.");
            System.out.println("Validation accuracy (" + String.format("%.4f", validationAccuracy) + 
                               ") is significantly higher than test accuracy (" + String.format("%.4f", testAccuracy) + ").");
            System.out.println("This could indicate that the validation set was not perfectly representative of the test set's distribution, or the model overfit slightly to the validation set during early stopping.");
        }
        
        if (trainFeatures.size() < 500) {
            System.out.println("Note: Training data size is relatively small (" + trainFeatures.size() + " records).");
            System.out.println("Smaller datasets can lead to less robust models and higher variance in results.");
        }
        if (testFeatures.size() < 100) {
            System.out.println("Note: Test data size is relatively small (" + testFeatures.size() + " records).");
            System.out.println("Smaller test sets provide less reliable estimates of true generalization performance.");
        }
        
        System.out.println("\nAnalysis complete. Please review any warnings and consider adjustments.");
    }

    private static void scaleFeatures() {
        int numFeatures = trainFeatures.get(0).length;
        featureMeans = new double[numFeatures];
        featureStdDevs = new double[numFeatures];
        
        // Calculate means
        for (double[] features : trainFeatures) {
            for (int i = 0; i < numFeatures; i++) {
                featureMeans[i] += features[i];
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureMeans[i] /= trainFeatures.size();
        }
        
        // Calculate standard deviations
        for (double[] features : trainFeatures) {
            for (int i = 0; i < numFeatures; i++) {
                double diff = features[i] - featureMeans[i];
                featureStdDevs[i] += diff * diff;
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureStdDevs[i] = Math.sqrt(featureStdDevs[i] / trainFeatures.size());
            if (featureStdDevs[i] == 0) featureStdDevs[i] = 1; // Prevent division by zero
        }
        
        // Scale features
        scaleFeatureList(trainFeatures);
        scaleFeatureList(validationFeatures);
        scaleFeatureList(testFeatures);
    }
    
    private static void scaleFeatureList(List<double[]> features) {
        for (double[] featureVector : features) {
            for (int i = 0; i < featureVector.length; i++) {
                featureVector[i] = (featureVector[i] - featureMeans[i]) / featureStdDevs[i];
            }
        }
    }

    /**
     * Tournament selection with fitness-proportionate selection
     */
    private static TreeNode tournamentSelection(List<TreeNode> population, Map<TreeNode, Double> fitnessScores) {
        TreeNode best = null;
        double bestFitness = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;
        
        // Select TOURNAMENT_SIZE unique individuals
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < population.size(); i++) {
            indices.add(i);
        }
        
        // Shuffle deterministically based on current position
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            int j = i + random.nextInt(indices.size() - i);
            int temp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, temp);
        }
        
        // Select the best from the tournament
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            TreeNode candidate = population.get(indices.get(i));
            double fitness = fitnessScores.get(candidate);
            
            // Use index as tiebreaker for deterministic selection
            if (fitness > bestFitness || (fitness == bestFitness && indices.get(i) < bestIndex)) {
                best = candidate;
                bestFitness = fitness;
                bestIndex = indices.get(i);
            }
        }
        
        return best;
    }

    // --- TreeNode and related classes (nested for self-containment) ---

    private static abstract class TreeNode {
        public abstract double evaluate(double[] input);
        public abstract TreeNode deepCopy();
        public abstract String toString();
    }

    private static class ConstantNode extends TreeNode {
        private double value;

        public ConstantNode(double value) {
            this.value = value;
        }

        @Override
        public double evaluate(double[] input) {
            return value;
        }

        @Override
        public TreeNode deepCopy() {
            return new ConstantNode(this.value);
        }

        @Override
        public String toString() {
            return String.format("%.2f", value);
        }
    }

    private static class FeatureNode extends TreeNode {
        private int featureIndex;

        public FeatureNode(int featureIndex) {
            this.featureIndex = featureIndex;
        }

        @Override
        public double evaluate(double[] input) {
            return input[featureIndex];
        }

        @Override
        public TreeNode deepCopy() {
            return new FeatureNode(this.featureIndex);
        }

        @Override
        public String toString() {
            return "F" + featureIndex;
        }
    }

    private static abstract class BinaryNode extends TreeNode {
        protected TreeNode left;
        protected TreeNode right;

        public BinaryNode(TreeNode left, TreeNode right) {
            this.left = left;
            this.right = right;
        }
    }

    private static class AddNode extends BinaryNode {
        public AddNode(TreeNode left, TreeNode right) {
            super(left, right);
        }

        @Override
        public double evaluate(double[] input) {
            return left.evaluate(input) + right.evaluate(input);
        }

        @Override
        public TreeNode deepCopy() {
            return new AddNode(left.deepCopy(), right.deepCopy());
        }

        @Override
        public String toString() {
            return "(" + left.toString() + " + " + right.toString() + ")";
        }
    }

    private static class SubtractNode extends BinaryNode {
        public SubtractNode(TreeNode left, TreeNode right) {
            super(left, right);
        }

        @Override
        public double evaluate(double[] input) {
            return left.evaluate(input) - right.evaluate(input);
        }

        @Override
        public TreeNode deepCopy() {
            return new SubtractNode(left.deepCopy(), right.deepCopy());
        }

        @Override
        public String toString() {
            return "(" + left.toString() + " - " + right.toString() + ")";
        }
    }

    private static class MultiplyNode extends BinaryNode {
        public MultiplyNode(TreeNode left, TreeNode right) {
            super(left, right);
        }

        @Override
        public double evaluate(double[] input) {
            return left.evaluate(input) * right.evaluate(input);
        }

        @Override
        public TreeNode deepCopy() {
            return new MultiplyNode(left.deepCopy(), right.deepCopy());
        }

        @Override
        public String toString() {
            return "(" + left.toString() + " * " + right.toString() + ")";
        }
    }

    private static class SafeDivideNode extends BinaryNode {
        public SafeDivideNode(TreeNode left, TreeNode right) {
            super(left, right);
        }

        @Override
        public double evaluate(double[] input) {
            double denominator = right.evaluate(input);
            if (Math.abs(denominator) < 1e-6) { // Avoid division by zero
                return left.evaluate(input); // Or 1.0, or 0.0, depending on desired behavior
            }
            return left.evaluate(input) / denominator;
        }

        @Override
        public TreeNode deepCopy() {
            return new SafeDivideNode(left.deepCopy(), right.deepCopy());
        }

        @Override
        public String toString() {
            return "(" + left.toString() + " / " + right.toString() + ")";
        }
    }

    private static class IfNode extends TreeNode {
        private TreeNode condition;
        private TreeNode ifTrue;
        private TreeNode ifFalse;

        public IfNode(TreeNode condition, TreeNode ifTrue, TreeNode ifFalse) {
            this.condition = condition;
            this.ifTrue = ifTrue;
            this.ifFalse = ifFalse;
        }

        @Override
        public double evaluate(double[] input) {
            if (condition.evaluate(input) >= 0) { // Condition is true if >= 0
                return ifTrue.evaluate(input);
            } else {
                return ifFalse.evaluate(input);
            }
        }

        @Override
        public TreeNode deepCopy() {
            return new IfNode(condition.deepCopy(), ifTrue.deepCopy(), ifFalse.deepCopy());
        }

        @Override
        public String toString() {
            return "IF(" + condition.toString() + ", " + ifTrue.toString() + ", " + ifFalse.toString() + ")";
        }
    }

    private static void initializeResultsFile() {
        try {
            java.io.File file = new java.io.File(RESULTS_FILE);
            resultsFileExists = file.exists();
            
            if (!resultsFileExists) {
                FileWriter writer = new FileWriter(RESULTS_FILE);
                writer.write("Seed,Training_Accuracy,Training_Precision,Training_Recall,Training_F1," +
                           "Validation_Accuracy,Validation_Precision,Validation_Recall,Validation_F1," +
                           "Test_Accuracy,Test_Precision,Test_Recall,Test_F1," +
                           "Training_TP,Training_FP,Training_TN,Training_FN," +
                           "Validation_TP,Validation_FP,Validation_TN,Validation_FN," +
                           "Test_TP,Test_FP,Test_TN,Test_FN\n");
                writer.close();
            }
        } catch (IOException e) {
            System.err.println("Error initializing results file: " + e.getMessage());
        }
    }

    private static void saveResults(double trainAccuracy, double validationAccuracy, double testAccuracy, TreeNode model) {
        try {
            FileWriter writer = new FileWriter(RESULTS_FILE, true);
            Locale.setDefault(Locale.US);  // Set US locale for decimal points
            
            // Get detailed metrics for each set
            Map<String, Object> trainMetrics = getDetailedMetrics(trainFeatures, trainLabels, model);
            Map<String, Object> validationMetrics = getDetailedMetrics(validationFeatures, validationLabels, model);
            Map<String, Object> testMetrics = getDetailedMetrics(testFeatures, testLabels, model);
            
            // Write results in CSV format using US locale
            writer.write(String.format(Locale.US, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                currentSeed,
                trainMetrics.get("accuracy"), trainMetrics.get("precision"), trainMetrics.get("recall"), trainMetrics.get("f1"),
                validationMetrics.get("accuracy"), validationMetrics.get("precision"), validationMetrics.get("recall"), validationMetrics.get("f1"),
                testMetrics.get("accuracy"), testMetrics.get("precision"), testMetrics.get("recall"), testMetrics.get("f1"),
                trainMetrics.get("tp"), trainMetrics.get("fp"), trainMetrics.get("tn"), trainMetrics.get("fn"),
                validationMetrics.get("tp"), validationMetrics.get("fp"), validationMetrics.get("tn"), validationMetrics.get("fn"),
                testMetrics.get("tp"), testMetrics.get("fp"), testMetrics.get("tn"), testMetrics.get("fn")
            ));
            
            writer.close();
            System.out.println("\nResults saved to " + RESULTS_FILE);
        } catch (IOException e) {
            System.err.println("Error saving results: " + e.getMessage());
        }
    }

    private static Map<String, Object> getDetailedMetrics(List<double[]> features, List<Integer> labels, TreeNode model) {
        int truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
        
        for (int i = 0; i < features.size(); i++) {
            double[] input = features.get(i);
            int actualClass = labels.get(i);
            
            double output = model.evaluate(input);
            int predictedClass = output >= 0 ? 1 : 0;
            
            if (predictedClass == 1 && actualClass == 1) truePositives++;
            else if (predictedClass == 1 && actualClass == 0) falsePositives++;
            else if (predictedClass == 0 && actualClass == 0) trueNegatives++;
            else falseNegatives++;
        }
        
        double accuracy = (double) (truePositives + trueNegatives) / (truePositives + falsePositives + trueNegatives + falseNegatives);
        double precision = truePositives == 0 ? 0 : (double) truePositives / (truePositives + falsePositives);
        double recall = truePositives == 0 ? 0 : (double) truePositives / (truePositives + falseNegatives);
        double f1Score = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
        
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("accuracy", accuracy);
        metrics.put("precision", precision);
        metrics.put("recall", recall);
        metrics.put("f1", f1Score);
        metrics.put("tp", truePositives);
        metrics.put("fp", falsePositives);
        metrics.put("tn", trueNegatives);
        metrics.put("fn", falseNegatives);
        
        return metrics;
    }
}