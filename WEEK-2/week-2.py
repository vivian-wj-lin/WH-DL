import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other_point):
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def __str__(self):
        return f"({self.x}, {self.y})"


class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def slope(self):
        if self.point2.x - self.point1.x == 0:
            return float("inf")
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    def is_parallel(self, other_line):
        return self.slope() == other_line.slope()

    def is_perpendicular(self, other_line):
        slope1 = self.slope()
        slope2 = other_line.slope()

        # 垂直線
        if slope1 == float("inf") and slope2 == 0:
            return True
        if slope1 == 0 and slope2 == float("inf"):
            return True
        if slope1 == float("inf") or slope2 == float("inf"):
            return False

        # perpendicular: slope1*slope2 = -1
        return abs(slope1 * slope2 + 1) < 0.0001

    def __str__(self):
        return f"Line from {self.point1} to {self.point2}"


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def area(self):
        return math.pi * self.radius**2

    def intersects(self, other_circle):
        distance = self.center.distance_to(other_circle.center)
        return distance < (self.radius + other_circle.radius) and distance > abs(
            self.radius - other_circle.radius
        )

    def __str__(self):
        return f"Circle at {self.center} with radius {self.radius}"


class Polygon:
    def __init__(self, points):
        self.points = points

    def perimeter(self):
        total = 0
        n = len(self.points)

        for i in range(n):
            next_i = (i + 1) % n
            total += self.points[i].distance_to(self.points[next_i])

        return total

    def __str__(self):
        points_str = ", ".join(str(p) for p in self.points)
        return f"Polygon with vertices: {points_str}"


def task1():
    # Line A
    point_a1 = Point(-6, 1)
    point_a2 = Point(2, 4)

    # Line B
    point_b1 = Point(-6, -1)
    point_b2 = Point(2, 2)

    # Line C
    point_c1 = Point(-4, -4)
    point_c2 = Point(-1, 6)

    polygon_points = [Point(-1, -2), Point(2, 0), Point(5, -1), Point(4, -4)]

    circle_a_center = Point(6, 3)
    circle_b_center = Point(8, 1)

    line_a = Line(point_a1, point_a2)
    line_b = Line(point_b1, point_b2)
    line_c = Line(point_c1, point_c2)

    circle_a = Circle(circle_a_center, 2)
    circle_b = Circle(circle_b_center, 1)

    polygon_a = Polygon(polygon_points)

    is_parallel = line_a.is_parallel(line_b)
    print("Are Line A and Line B parallel?", is_parallel)

    is_perpendicular = line_c.is_perpendicular(line_a)
    print("Are Line C and Line A perpendicular?", is_perpendicular)

    area_a = circle_a.area()
    print(f"Print the area of Circle A. {area_a:.4f}")

    do_intersect = circle_a.intersects(circle_b)
    print("Do Circle A and Circle B intersect?", do_intersect)

    perimeter_a = polygon_a.perimeter()
    print(f"Print the perimeter of Polygon A. {perimeter_a:.4f}")


class Enemy:
    def __init__(self, label, position, move_vector):
        self.label = label
        self.position = position
        self.move_vector = move_vector
        self.life_points = 10
        self.is_alive = True

    def move(self):
        if self.is_alive:
            self.position.x += self.move_vector[0]
            self.position.y += self.move_vector[1]

    def take_damage(self, damage):
        if self.is_alive:
            self.life_points -= damage
            if self.life_points <= 0:
                self.is_alive = False

    def __str__(self):
        status = "alive" if self.is_alive else "dead"
        return f"{self.label}: position{self.position}, life points={self.life_points}, status={status}"


class Tower:
    def __init__(self, label, position, attack_points, attack_range):
        self.label = label
        self.position = position
        self.attack_points = attack_points
        self.attack_range = attack_range

    def is_in_range(self, enemy):
        if not enemy.is_alive:
            return False
        distance = self.position.distance_to(enemy.position)
        return distance <= self.attack_range

    def attack(self, enemies):
        attacked = []
        for enemy in enemies:
            if self.is_in_range(enemy):
                enemy.take_damage(self.attack_points)
                attacked.append(enemy.label)
        return attacked

    def __str__(self):
        return f"{self.label}: position{self.position}, attack points={self.attack_points}, attack range={self.attack_range}"


class BasicTower(Tower):
    def __init__(self, label, position):
        super().__init__(label, position, attack_points=1, attack_range=2)


class AdvancedTower(Tower):
    def __init__(self, label, position):
        super().__init__(label, position, attack_points=2, attack_range=4)


def task2():
    enemies = [
        Enemy("E1", Point(-10, 2), (2, -1)),
        Enemy("E2", Point(-8, 0), (3, 1)),
        Enemy("E3", Point(-9, -1), (3, 0)),
    ]

    basic_towers = [
        BasicTower("T1", Point(-3, 2)),
        BasicTower("T2", Point(-1, -2)),
        BasicTower("T3", Point(4, 2)),
        BasicTower("T4", Point(7, 0)),
    ]

    advanced_towers = [
        AdvancedTower("A1", Point(1, 1)),
        AdvancedTower("A2", Point(4, -3)),
    ]

    all_towers = basic_towers + advanced_towers

    for turn in range(1, 11):
        for enemy in enemies:
            if enemy.is_alive:
                enemy.move()

        for tower in all_towers:
            tower.attack(enemies)

    for enemy in enemies:
        print(
            f"{enemy.label}: final position={enemy.position}, life points={enemy.life_points}"
        )


if __name__ == "__main__":
    task1()
    print("=" * 20)
    task2()
