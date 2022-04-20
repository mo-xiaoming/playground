package net.skelix.sbtb.dao;

import net.skelix.sbtb.model.Person;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository("fakeDao")
public class FakePersonDataAccessService implements PersonDao {
    private static final List<Person> DB = new ArrayList<>();

    @Override
    public int insertPerson(UUID id, Person person) {
        DB.add(new Person(id, person.getName()));
        return 1;
    }

    @Override
    public List<Person> selectAllPeople() {
        return DB;
    }

    @Override
    public Optional<Person> selectPersonById(UUID id) {
        return DB.stream().filter(p -> p.getId().equals(id)).findAny();
    }

    @Override
    public int deletePersonById(UUID id) {
        return DB.removeIf(p -> p.getId().equals(id)) ? 1 : 0;
    }

    @Override
    public int updatePersonById(UUID id, Person person) {
        var p = selectPersonById(id);
        if (p.isPresent()) {
            final var i = DB.indexOf(p.get());
            DB.set(i, person);
            return 1;
        }
        return 0;
    }
}
