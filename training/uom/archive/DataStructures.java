import java.util.*;

public class DataStructures {

    public static void main(String[] args) {
        
        // LinkedList<String> list = new LinkedList<String>();

        HashSet<String> set = new HashSet<>();

        set.add("John");
        set.add("Nick");
        set.add("Mary");
        set.add("John");

        for (String name: set) 
            System.out.println(name);
    }
    
}
 