import java.util.ArrayList;

public class Ship {
    
    private String name;
    private int capacity;
    private ArrayList<Container> containers = new ArrayList<Container>();

    public Ship(String name, int capacity) {

        this.name = name;
        this.capacity = capacity;
    }
    
    public void addContainer(Container container) {

        if (containers.size() < capacity) {
            containers.add(container);
        } else {
            System.out.println("Sorry, the ship is full.");
        }
    }

    public double getTotalCharge() {

        double charge = 0;

        for (Container container: containers) {
            charge += container.getCharge();
        }
        return charge;
    }

    public String getName() {
        return name;
    }

    public int getShipLoad() {
        return containers.size();
    }
}
