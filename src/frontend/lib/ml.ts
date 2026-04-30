import LogisticRegression from 'ml-logistic-regression';
import { Matrix } from 'ml-matrix';
import { getFlattenedData } from './db.js';

export class PurchaseModel {
  private model: LogisticRegression | null = null;

  private segmentMap: Record<string, number> = { low: 0, medium: 1, high: 2 };
  private timeOfDayMap: Record<string, number> = { morning: 0, afternoon: 1, evening: 2 };

  async train() {
    const data = getFlattenedData();

    const X: number[][] = [];
    const y: number[] = [];

    data.forEach(row => {
      X.push(this.preprocessRow(row));
      y.push(row.bought ? 1 : 0);
    });

    // 👉 Convert to Matrix
    const XMatrix = new Matrix(X);
    const yMatrix = Matrix.columnVector(y);

    this.model = new LogisticRegression({
      numSteps: 1000,
      learningRate: 5e-3
    });

    this.model.train(XMatrix, yMatrix);

    console.log('Model trained on', data.length, 'samples');
  }

  private preprocessRow(row: any): number[] {
    return [
      row.response_time_min,
      row.price,
      row.patience_level,
      this.segmentMap[row.segment] ?? 0,
      this.timeOfDayMap[row.time_of_day] ?? 0,
      row.complexity
    ];
  }

  predict(input: {
    response_time_min: number,
    price: number,
    patience_level: number,
    segment: 'low' | 'medium' | 'high',
    time_of_day: 'morning' | 'afternoon' | 'evening',
    complexity: number
  }): number {
    if (!this.model) {
      throw new Error('Model not trained');
    }

    const preprocessed = this.preprocessRow(input);
    const inputMatrix = new Matrix([preprocessed]);

    const probabilities = this.model.predict(inputMatrix);

    // ⚠️ This lib returns predictions differently than expected
    return probabilities[0];
  }
}

export const instance = new PurchaseModel();