import java.util.ArrayList;
import java.util.Collections;

public class Main {

    public static void main(String[] args) {

        ArrayList<String> names = new ArrayList<String>();

        names.add("Helen");
        names.add("Bob");
        names.add("Nick");
        names.add("Mike");
        names.add("Mary");

        System.out.println("\n-------Sorted-------\n");
        Collections.sort(names);
        System.out.println(names);

        System.out.println("\n-------Reverse-------\n");
        Collections.reverse(names);
        System.out.println(names);

        System.out.println("\n-------Swap--------\n");
        Collections.swap(names, 2, 3);
        System.out.println(names);

        System.out.println("\n-----Max Min-------\n");
        String max = Collections.max(names);
        String min = Collections.min(names);
        System.out.println("Max: " + max + ", Min: " + min);

        System.out.println("\n-------Frequency-------\n");
        names.add("Helen");
        names.add("Helen");
        System.out.println(Collections.frequency(names, "Helen"));

        System.out.println("\n-------Shuffled-------\n");
        Collections.shuffle(names);

        System.out.println(names);
    } 
    
    
}
