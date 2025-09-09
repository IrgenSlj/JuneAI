public class DataSet {

    private int counter;
    private double sum;
    private Measurable min;
    private Measurable max;
    
    public DataSet() {
        counter = 0;
        sum = 0;
        max = null;
        min = null;
    }

    public void add(Measurable item) {
        if (counter == 0) {
            min = item;
            max = item;
        } else if (item.getMeasure() > max.getMeasure()) {
            max = item;
        } else if (item.getMeasure() < min.getMeasure()) {
            min = item;
        }

        counter++;
        sum += item.getMeasure();
    }

    public double calcAverage() {
        if (counter == 0) 
            return 0;
        return sum / counter;
    }

    public Measurable getMin() {
        return min;
    }

    public Measurable getMax() {
        return max;
    }

}
