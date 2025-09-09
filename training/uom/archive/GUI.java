import java.awt.event.*;
import java.util.*;
import javax.swing.*;

public class GUI extends JFrame {
    
    private JTextField nameField;
    private JButton addButton;
    private JList list;
    private JButton sortButton;
    private JPanel panel;
    private DefaultListModel model;
    private JScrollPane scrollPane;

    public GUI() {

        nameField = new JTextField(10);
        addButton = new JButton("add");
        list = new JList<>();
        sortButton = new JButton("sort");
        panel = new JPanel();
        model = new DefaultListModel<>();
        list.setModel(model);
        scrollPane = new JScrollPane(list);

        panel.add(nameField);
        panel.add(addButton);
        panel.add(scrollPane);
        panel.add(sortButton);

        this.setContentPane(panel);

        addButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                String name = nameField.getText();
                model.addElement(name);
            }
        });

        sortButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                ArrayList<String> names = Collections.list(model.elements());
                Collections.sort(names);
                model.clear();

                for (String name: names)
                    model.addElement(name);

            }
        }
        
        );
        this.setVisible(true);
        this.setSize(500, 500);
    }
}
