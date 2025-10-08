import numpy as np

BOUNDING_BOX_DISTANCE = 1


def point_to_line_distance(point, line_start, line_end):
    """
    Berechnet den kürzesten Abstand eines Punktes zu einer Linie (definiert durch zwei Punkte).
    """
    line_vec = line_end - line_start
    line_length_sq = np.dot(line_vec, line_vec)

    if line_length_sq == 0:
        # Linie ist ein Punkt
        return np.linalg.norm(point - line_start)

    # Projektion des Punktes auf die Linie
    t = np.dot(point - line_start, line_vec) / line_length_sq
    t = max(0, min(1, t))  # Begrenze t auf [0,1] für Segment

    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def is_point_in_pipe_bounding_box(point, pipe_p1, pipe_p2, distance):
    """
    Prüft ob ein Punkt innerhalb der Bounding Box eines Rohres liegt.
    Die Bounding Box ist der Bereich um das Rohrsegment mit dem gegebenen Abstand.
    """
    return point_to_line_distance(point, pipe_p1, pipe_p2) <= distance


def is_individual_valid(ind, found_pipes, pts):
    """
    Prüft ob ein Individuum gültig ist (nicht beide Punkte in derselben Pipe-Bounding-Box).
    """
    if len(found_pipes) == 0:
        return True

    p1, p2 = pts[ind[0]], pts[ind[1]]

    for pipe in found_pipes:
        pipe_p1 = np.array([pipe["p1_x"], pipe["p1_y"], pipe["p1_z"]])
        pipe_p2 = np.array([pipe["p2_x"], pipe["p2_y"], pipe["p2_z"]])

        # Prüfe ob BEIDE Punkte des Individuums in der Bounding Box dieser Pipe sind
        p1_in_box = is_point_in_pipe_bounding_box(
            p1, pipe_p1, pipe_p2, BOUNDING_BOX_DISTANCE
        )
        p2_in_box = is_point_in_pipe_bounding_box(
            p2, pipe_p1, pipe_p2, BOUNDING_BOX_DISTANCE
        )

        if p1_in_box and p2_in_box:
            return False  # Ungültig, da beide Punkte in derselben Pipe-Box

    return True
