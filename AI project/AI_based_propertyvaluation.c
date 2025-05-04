#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 5        // Number of training samples
#define FEATURES 5 // Area, Bedrooms, Bathrooms, Age, Location Score

// Training data: area, bedrooms, bathrooms, age, location_score
float X[N][FEATURES] = {
    {1000, 2, 1, 10, 3},
    {1500, 3, 2, 5, 6},
    {2000, 3, 2, 3, 8},
    {2500, 4, 3, 8, 9},
    {3000, 4, 3, 2, 10}
};

float y[N] = {50, 80, 100, 120, 150}; // Price in Lakhs
float weights[FEATURES + 1];          // +1 for bias term

// Function to map location string to score
float get_location_score(const char* location) {
    if (strcmp(location, "rural") == 0) return 3;
    if (strcmp(location, "suburban") == 0) return 6;
    if (strcmp(location, "urban") == 0) return 9;
    if (strcmp(location, "metro") == 0) return 10;
    return 5; // default score
}

// Mean of array
float mean(float arr[], int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += arr[i];
    return sum / size;
}

// Train using normal equation (approximation)
void train_model() {
    float x_mean[FEATURES] = {0};
    float y_mean_val = mean(y, N);

    for (int j = 0; j < FEATURES; j++) {
        for (int i = 0; i < N; i++) {
            x_mean[j] += X[i][j];
        }
        x_mean[j] /= N;
    }

    for (int j = 0; j < FEATURES; j++) {
        float numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < N; i++) {
            numerator += (X[i][j] - x_mean[j]) * (y[i] - y_mean_val);
            denominator += (X[i][j] - x_mean[j]) * (X[i][j] - x_mean[j]);
        }
        weights[j + 1] = denominator != 0 ? (numerator / denominator) : 0;
    }

    float bias = y_mean_val;
    for (int j = 0; j < FEATURES; j++) {
        bias -= weights[j + 1] * x_mean[j];
    }
    weights[0] = bias;
}

// Predict price based on input
float predict(float area, int bedrooms, int bathrooms, int age, float location_score) {
    float x[] = {area, (float)bedrooms, (float)bathrooms, (float)age, location_score};
    float prediction = weights[0]; // bias
    for (int i = 0; i < FEATURES; i++) {
        prediction += weights[i + 1] * x[i];
    }
    return prediction;
}

// Entry point
int main() {
    train_model();

    float area;
    int bedrooms, bathrooms, age;
    char location[20];
    float location_score;
    char choice;

    printf("🏠 AI-Based Property Valuation System (C Version)\n");

REENTER_INPUT:
    printf("\nEnter area in sq ft: ");
    scanf("%f", &area);
    printf("Enter number of bedrooms: ");
    scanf("%d", &bedrooms);
    printf("Enter number of bathrooms: ");
    scanf("%d", &bathrooms);
    printf("Enter age of property (in years): ");
    scanf("%d", &age);
    printf("Enter location type (rural / suburban / urban / metro): ");
    scanf("%s", location);

    location_score = get_location_score(location);

    // Show entered data
    printf("\n You entered:\n");
    printf("Area: %.2f sq ft\n", area);
    printf("Bedrooms: %d\n", bedrooms);
    printf("Bathrooms: %d\n", bathrooms);
    printf("Age: %d years\n", age);
    printf("Location: %s (score: %.1f)\n", location, location_score);

    // Ask if user wants to change anything
    printf("\n Do you want to change any value? (y/n): ");
    scanf(" %c", &choice); // space before %c to consume newline

    if (choice == 'y' || choice == 'Y') {
        printf("\nWhich value do you want to change?\n");
        printf("1. Area\n2. Bedrooms\n3. Bathrooms\n4. Age\n5. Location\n");
        printf("Enter number (or 0 to re-enter all): ");
        int edit_choice;
        scanf("%d", &edit_choice);

        switch (edit_choice) {
            case 1:
                printf("Enter new area: ");
                scanf("%f", &area);
                break;
            case 2:
                printf("Enter new number of bedrooms: ");
                scanf("%d", &bedrooms);
                break;
            case 3:
                printf("Enter new number of bathrooms: ");
                scanf("%d", &bathrooms);
                break;
            case 4:
                printf("Enter new age: ");
                scanf("%d", &age);
                break;
            case 5:
                printf("Enter new location: ");
                scanf("%s", location);
                location_score = get_location_score(location);
                break;
            case 0:
                goto REENTER_INPUT;
            default:
                printf(" Invalid choice. Continuing with previous values.\n");
        }
    }

    float price = predict(area, bedrooms, bathrooms, age, location_score);

printf("\n-------------------------------\n");
printf("  Feature         | Value\n");
printf("-------------------------------\n");
printf("  Area            | %.2f sq ft\n", area);
printf("  Bedrooms        | %d\n", bedrooms);
printf("  Bathrooms       | %d\n", bathrooms);
printf("  Age             | %d years\n", age);
printf("  Location        | %s (score: %.1f)\n", location, location_score);
printf("-------------------------------\n");
printf("  Estimated Price | Rupees %.2f Lakhs\n", price);
printf("-------------------------------\n");

    return 0;
}
