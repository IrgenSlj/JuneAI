public class BulkContainer extends Container {
    
    private double weight;

    public BulkContainer(String code, String destination, double weight) {
        super(code, destination);
        this.weight = weight;
    }

    public double getCharge() {
        return 10*weight;
    }
}
